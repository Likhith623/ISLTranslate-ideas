"""Complete landmark-to-text network; PCAT is the only temporal compressor."""
import torch
from torch import Tensor, nn
from .config import ModelConfig
from .geometry import BodyFrame
from .operators import ProgressCoupledAreaTransport, relative_times
from .tokenizer import ByteTokenizer, Utf8Constraint
from .vision import CropEncoder
from .moving_frame import MovingBodyFrame


class SignTranslator(nn.Module):
    def __init__(self, cfg: ModelConfig, initialize_pretrained=True, text_codec_payload=None):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.frame = BodyFrame(cfg.anchors, cfg.anchor_min_confidence, cfg.anchor_policy)
        # Affine region maps preserve linear interpolation, unlike a pointwise MLP.
        self.region_maps = nn.ModuleList([nn.Linear(4 * len(r), cfg.rank) for r in cfg.regions])
        self.operator = ProgressCoupledAreaTransport(
            len(cfg.regions) * cfg.rank, cfg.bins, cfg.width, cfg.metric_floor,
            cfg.clock, cfg.operator, cfg.timing, cfg.progress_scales,
            cfg.progress_augmentation, cfg.metric_condition_limit, cfg.integration, cfg.integration_precision,
            cfg.transport, cfg.area_sketch_dim, cfg.area_sketch_heads, cfg.adaptive_step, cfg.adaptive_min_bins)
        self.source_position = nn.Parameter(torch.empty(1, self.operator.token_count, cfg.width))
        self.target_position = nn.Parameter(torch.empty(1, cfg.max_target_length, cfg.width)) if cfg.text_backend == "byte" else None
        nn.init.normal_(self.source_position, std=0.02)
        if self.target_position is not None:
            nn.init.normal_(self.target_position, std=0.02)
        self.local_operators = nn.ModuleList()
        if cfg.local_clocks:
            self.local_operators.extend([ProgressCoupledAreaTransport(cfg.rank, cfg.local_bins, cfg.width,
                timing=cfg.timing, augment_progress=cfg.progress_augmentation, transport=cfg.transport,
                integration_precision=cfg.integration_precision) for _ in cfg.regions])
            self.local_position = nn.Parameter(torch.zeros(1, len(cfg.regions) * cfg.local_bins, cfg.width))
        self.moving_frame = None
        if cfg.body_frame == "dual":
            self.moving_frame = MovingBodyFrame(cfg.anchors, cfg.anchor_min_confidence, cfg.body_time_constant)
            self.motion_projection = nn.Linear(11, cfg.width)
        self.vision = None
        if cfg.visual_backend != "none":
            self.vision = CropEncoder(cfg, initialize_pretrained)
            self.visual_to_pose = nn.Linear(cfg.visual_dim, cfg.rank)
            self.visual_gain = nn.Parameter(torch.zeros(len(cfg.regions)))
            self.visual_to_memory = nn.Linear(cfg.visual_dim, cfg.width)
            self.visual_normalize = nn.LayerNorm(cfg.width) if cfg.visual_memory_norm else nn.Identity()
            self.visual_identity = nn.Parameter(torch.randn(cfg.visual_regions, cfg.width) * .02)
        encoder_layer = nn.TransformerEncoderLayer(
            cfg.width, cfg.heads, cfg.width * cfg.ff_multiplier, cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.encoder_layers,
                                             norm=nn.LayerNorm(cfg.width), enable_nested_tensor=False)
        self.pretrained_decoder = None
        self.codec = ByteTokenizer()
        if cfg.text_backend == "hf":
            from .text import PretrainedDecoder
            self.pretrained_decoder = PretrainedDecoder(cfg, initialize_pretrained, text_codec_payload)
            self.codec = self.pretrained_decoder.codec
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                cfg.width, cfg.heads, cfg.width * cfg.ff_multiplier, cfg.dropout,
                activation="gelu", batch_first=True, norm_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer, cfg.decoder_layers, norm=nn.LayerNorm(cfg.width))
            self.embedding = nn.Embedding(ByteTokenizer.vocab_size, cfg.width, padding_idx=0)
            self.output = nn.Linear(cfg.width, ByteTokenizer.vocab_size, bias=False)

    def canonical_features(self, pose: Tensor, confidence: Tensor, lengths: Tensor) -> Tensor:
        if pose.ndim != 4 or pose.shape[2:] != (self.cfg.joints, 3):
            raise ValueError("pose layout does not match config")
        if confidence.shape != pose.shape[:-1]:
            raise ValueError("confidence shape mismatch")
        if self.cfg.depth_policy == "ignore":
            pose = torch.cat((pose[..., :2], torch.zeros_like(pose[..., 2:])), -1)
        b, t = pose.shape[:2]
        if lengths.shape != (b,) or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("lengths must be an integer vector [B]")
        if ((lengths < 2) | (lengths > t)).any():
            raise ValueError("invalid sequence lengths")
        valid = torch.arange(t, device=pose.device)[None] < lengths[:, None]
        if not torch.isfinite(pose[valid]).all() or not torch.isfinite(confidence[valid]).all():
            raise ValueError("valid positions must be finite; impute missing joints in the loader")
        if ((confidence[valid] < 0) | (confidence[valid] > 1)).any():
            raise ValueError("confidence outside [0,1]")
        pose = torch.where(valid[..., None, None], pose, 0)
        confidence = torch.where(valid[..., None], confidence, 0)
        if self.cfg.allow_unanchored_rgb:
            anchored = self.frame.eligible_frames(pose, confidence, lengths).any(1)
            coords = torch.zeros_like(pose)
            if anchored.any():
                coords[anchored] = self.frame(pose[anchored], confidence[anchored], lengths[anchored])
            confidence = torch.where(anchored[:, None, None], confidence, 0)
        else:
            coords = self.frame(pose, confidence, lengths)
        return torch.where(valid[..., None, None], torch.cat((coords, confidence[..., None]), dim=-1), 0)

    def embed_features(self, features):
        blocks = [mapping(features[:, :, region].flatten(2))
                  for mapping, region in zip(self.region_maps, self.cfg.regions)]
        return torch.cat(blocks, dim=-1)

    def embed_pose(self, pose: Tensor, confidence: Tensor, lengths: Tensor) -> Tensor:
        return self.embed_features(self.canonical_features(pose, confidence, lengths))

    def encode(self, pose: Tensor, confidence: Tensor, lengths: Tensor,
               timestamps: Tensor | None = None, rgb=None, rgb_mask=None) -> Tensor:
        """Return memory; use encode_with_mask when detail memory includes padding."""
        return self.encode_with_mask(pose, confidence, lengths, timestamps, rgb, rgb_mask)[0]

    def encode_invariant(self, pose, confidence, lengths, timestamps=None):
        if (self.cfg.timing or self.cfg.operator == "temporal" or self.vision is not None
                or self.moving_frame is not None or self.cfg.local_clocks or self.cfg.adaptive_step):
            raise ValueError("invariant encoding requires timing=false and progress cells")
        z = self.embed_pose(pose, confidence, lengths)
        tokens = self.operator(z, lengths, timestamps)
        return self.encoder(tokens + self.source_position)

    def encode_with_mask(self, pose, confidence, lengths, timestamps=None, rgb=None, rgb_mask=None):
        if self.cfg.timing and timestamps is None:
            raise ValueError("timing-enabled models require timestamps in seconds")
        features = self.canonical_features(pose, confidence, lengths)
        motion = None
        coarse_features = features
        if self.moving_frame is not None:
            local, motion = self.moving_frame(features, lengths, timestamps)
            coarse_features = torch.cat((local, features[..., 3:]), -1)
        z = self.embed_features(coarse_features)
        visual, visual_valid = None, None
        if self.vision is not None:
            if rgb is None or rgb_mask is None:
                raise ValueError("visual mode requires aligned RGB crops and observation masks")
            visual, visual_valid = self.vision(rgb, rgb_mask, lengths)
            if self.cfg.allow_unanchored_rgb:
                has_pose = features[..., 3].sum(dim=(1,2)) > 0
                if (~has_pose & ~visual_valid.any(dim=(1,2))).any():
                    raise ValueError("sample has neither anchored pose nor observed RGB")
            projected = self.visual_to_pose(visual) * visual_valid[..., None]
            blocks = list(z.split(self.cfg.rank, -1))
            for a in range(len(blocks)):
                indices = [i for i, region in enumerate(self.cfg.visual_to_articulator) if region == a]
                if indices:
                    count = visual_valid[:, :, indices].sum(-1, keepdim=True).clamp_min(1)
                    blocks[a] = blocks[a] + self.visual_gain[a].tanh() * projected[:, :, indices].sum(2) / count
            z = torch.cat(blocks, -1)
        tokens, mask, _ = self.operator.forward_with_mask(z, lengths, timestamps)
        tokens = tokens + self.source_position[:, :tokens.shape[1]]
        if self.cfg.local_clocks:
            local_tokens = [op(block, lengths, timestamps) for op, block in
                            zip(self.local_operators, z.split(self.cfg.rank, -1))]
            extra = torch.cat(local_tokens, 1) + self.local_position
            tokens = torch.cat((tokens, extra), 1)
            mask = torch.cat((mask, torch.zeros(extra.shape[:2], device=z.device, dtype=torch.bool)), 1)
        coarse = self.encoder(tokens, src_key_padding_mask=mask)
        memory, masks = [coarse], [mask]
        b, t = features.shape[:2]
        valid = torch.arange(t, device=pose.device)[None] < lengths[:, None]
        d = torch.where(valid[:, 1:, None], z[:, 1:] - z[:, :-1], 0)
        ell = torch.linalg.vector_norm(d * self.operator.metric.sqrt(), dim=-1)
        progress = torch.cat((z.new_zeros(b, 1), ell.cumsum(1)), dim=1)
        total = ell.sum(1, keepdim=True)
        progress = progress / torch.where(total > 0, total, 1)
        # The exact canonical vector is a reserved, unprojected subspace of
        # memory. Position and optional duration occupy disjoint coordinates.
        frequency = torch.arange(1, self.cfg.detail_position_dim // 2 + 1, device=z.device, dtype=z.dtype)
        phase = torch.pi * progress[..., None] * frequency
        position = torch.cat((phase.sin(), phase.cos()), -1)
        memory_position = torch.nn.functional.pad(position, (0, self.cfg.width - position.shape[-1]))
        if visual is not None or motion is not None:
            # A static pose/gated fusion can have zero geometric progress even
            # while RGB changes. Bind appearance/motion to an explicit frame
            # coordinate rather than turning those memories into an unordered set.
            if self.cfg.timing:
                if timestamps is None:
                    raise ValueError("timing-enabled multimodal memory needs timestamps")
                relative = relative_times(timestamps, z, valid)
                coordinate = relative / relative.gather(1, (lengths - 1)[:, None])
            else:
                coordinate = torch.arange(t, device=z.device, dtype=z.dtype)[None] / (lengths[:, None] - 1)
            time_phase = torch.pi * coordinate[..., None] * frequency
            time_position = torch.cat((time_phase.sin(), time_phase.cos()), -1)
            memory_position = memory_position + torch.nn.functional.pad(
                time_position, (0, self.cfg.width - time_position.shape[-1]))
        if visual is not None:
            v = self.visual_normalize(self.visual_to_memory(visual) + self.visual_identity[None, None]
                                      + memory_position[:, :, None])
            memory.append(torch.where(visual_valid[..., None], v, 0).flatten(1, 2))
            masks.append(~visual_valid.flatten(1, 2))
        if motion is not None:
            memory.append(self.motion_projection(motion) + memory_position)
            anchored_example = features[..., 3].sum(dim=(1, 2)) > 0
            masks.append(~valid | ~anchored_example[:, None])
        if not self.cfg.detail_memory:
            return torch.cat(memory, 1), torch.cat(masks, 1)
        if self.cfg.timing:
            if timestamps is None:
                raise ValueError("timing-enabled detail memory requires timestamps in seconds")
            ts = relative_times(timestamps, z, valid)
            duration = ts.gather(1, (lengths - 1)[:, None])
            timing = torch.stack((torch.log1p(ts.clamp_min(0)), ts / duration), -1)
        else:
            timing = z.new_zeros(b, t, 2)
        detail = torch.cat((features.flatten(2).to(z.dtype), position, timing), -1)
        detail = torch.nn.functional.pad(detail, (0, self.cfg.width - detail.shape[-1]))
        detail = torch.where(valid[..., None], detail, 0)
        memory.append(detail)
        masks.append(~valid)
        return torch.cat(memory, 1), torch.cat(masks, 1)

    def decode(self, decoder_input: Tensor, memory: Tensor, memory_padding_mask: Tensor | None = None) -> Tensor:
        length = decoder_input.shape[1]
        if length < 1 or length > self.cfg.max_target_length:
            raise ValueError("decoder length outside configured bounds")
        if self.pretrained_decoder is not None:
            if memory_padding_mask is None:
                memory_padding_mask = torch.zeros(memory.shape[:2], device=memory.device, dtype=torch.bool)
            return self.pretrained_decoder(decoder_input, memory, memory_padding_mask)
        hidden = self.embedding(decoder_input) + self.target_position[:, :length]
        causal = torch.ones(length, length, device=hidden.device, dtype=torch.bool).triu(1)
        hidden = self.decoder(hidden, memory, tgt_mask=causal,
                              memory_key_padding_mask=memory_padding_mask,
                              tgt_key_padding_mask=decoder_input.eq(ByteTokenizer.pad_id))
        return self.output(hidden)

    def forward(self, pose, confidence, lengths, decoder_input, timestamps=None, rgb=None, rgb_mask=None):
        memory, mask = self.encode_with_mask(pose, confidence, lengths, timestamps, rgb, rgb_mask)
        return self.decode(decoder_input, memory, mask)

    @torch.no_grad()
    def generate(self, pose, confidence, lengths, timestamps=None, max_new_tokens=None, rgb=None, rgb_mask=None):
        if self.training:
            raise RuntimeError("call model.eval() before deterministic generation")
        limit = self.cfg.max_target_length if max_new_tokens is None else max_new_tokens
        if type(limit) is not int or not 1 <= limit <= self.cfg.max_target_length:
            raise ValueError("invalid generation length")
        memory, memory_mask = self.encode_with_mask(pose, confidence, lengths, timestamps, rgb, rgb_mask)
        if self.pretrained_decoder is not None:
            return self.pretrained_decoder.generate(memory, memory_mask, limit, self.cfg.generation_beams)
        ids = torch.full((pose.shape[0], 1), ByteTokenizer.bos_id, dtype=torch.long, device=pose.device)
        finished = torch.zeros(pose.shape[0], device=pose.device, dtype=torch.bool)
        constraint = Utf8Constraint(pose.shape[0], pose.device)
        for step in range(limit):
            logits = self.decode(ids, memory, memory_mask)[:, -1]
            logits = constraint.mask(logits, limit - step)
            next_id = logits.argmax(-1)
            next_id = torch.where(finished, ByteTokenizer.pad_id, next_id)
            ids = torch.cat((ids, next_id[:, None]), dim=1)
            finished = finished | next_id.eq(ByteTokenizer.eos_id)
            constraint.advance(next_id, finished)
            if finished.all():
                break
        return ids
