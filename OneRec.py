import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Config:
    PAD_TOKEN_ID = 0
    BOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2

    VOCAB_SIZE = 3072 + 3 
    TARGET_SEQ_LEN = 3

    # TODO: test input
    # pathway 1
    USER_STATIC_DIM = 64
    # pathway 2
    SHORT_TERM_LEN = 20
    # pathway 3
    POS_FEEDBACK_LEN = 256
    # pathway 4
    LIFELONG_LEN = 128
    # 所有行为序列的特征维度简化为同一个值
    BEHAVIOR_FEAT_DIM = 128

    # 模型
    D_MODEL = 256
    N_HEAD = 8
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# encoder
class OneRecEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # TODO 4个pathway的输入处理，这里简化了，需要根据实际数据进行调整
        self.user_static_proj = nn.Linear(config.USER_STATIC_DIM, config.D_MODEL)        
        self.short_term_proj = nn.Linear(config.BEHAVIOR_FEAT_DIM, config.D_MODEL)
        self.pos_feedback_proj = nn.Linear(config.BEHAVIOR_FEAT_DIM, config.D_MODEL)
        self.lifelong_proj = nn.Linear(config.BEHAVIOR_FEAT_DIM, config.D_MODEL)
        
        self.pos_encoder = PositionalEncoding(config.D_MODEL, config.DROPOUT)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL, 
            nhead=config.N_HEAD,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_ENCODER_LAYERS)

    def forward(self, user_static, short_term, positive_feedback, lifelong):
        static_emb = self.user_static_proj(user_static).unsqueeze(1)
        short_emb = self.short_term_proj(short_term)
        pos_emb = self.pos_feedback_proj(positive_feedback)
        lifelong_emb = self.lifelong_proj(lifelong)
        
        combined_input = torch.cat([static_emb, short_emb, pos_emb, lifelong_emb], dim=1)
        
        memory = self.transformer_encoder(combined_input)
        
        return memory

# decoder
class OneRecDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.VOCAB_SIZE, config.D_MODEL, padding_idx=config.PAD_TOKEN_ID)
        self.pos_encoder = PositionalEncoding(config.D_MODEL, config.DROPOUT)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.D_MODEL,
            nhead=config.N_HEAD,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.NUM_DECODER_LAYERS)
        self.output_proj = nn.Linear(config.D_MODEL, config.VOCAB_SIZE)
    
    def forward(self, tgt_ids, memory, tgt_mask=None, tgt_padding_mask=None):
        tgt_emb = self.pos_encoder(self.token_embedding(tgt_ids))
        
        output = self.transformer_decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        logits = self.output_proj(output)
        return logits



class OneRec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = OneRecEncoder(config)
        self.decoder = OneRecDecoder(config)

    def forward(self, user_static, short_term, positive_feedback, lifelong, tgt_ids, tgt_mask, tgt_padding_mask):
        memory = self.encoder(user_static, short_term, positive_feedback, lifelong)
        logits = self.decoder(tgt_ids, memory, tgt_mask, tgt_padding_mask)
        return logits

    @staticmethod
    def _generate_square_subsequent_mask(sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    @torch.no_grad()
    def generate(self, user_static, short_term, positive_feedback, lifelong, max_len=None):
        if max_len is None:
            max_len = self.config.TARGET_SEQ_LEN
            
        self.eval()
        device = user_static.device
        
        memory = self.encoder(user_static, short_term, positive_feedback, lifelong)
        
        tgt_ids = torch.full((memory.size(0), 1), self.config.BOS_TOKEN_ID, dtype=torch.long, device=device)
        
        for _ in range(max_len):
            tgt_len = tgt_ids.size(1)
            tgt_mask = self._generate_square_subsequent_mask(tgt_len).to(device)
            logits = self.decoder(tgt_ids, memory, tgt_mask=tgt_mask)
            
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            
            tgt_ids = torch.cat([tgt_ids, next_token_id.unsqueeze(1)], dim=1)

            if (next_token_id == self.config.EOS_TOKEN_ID).all():
                break

        return tgt_ids[:, 1:]

# 测试数据
def create_dummy_dataset(batch_size, config, device):
    # Encoder inputs
    user_static = torch.randn(batch_size, config.USER_STATIC_DIM, device=device)
    short_term = torch.randn(batch_size, config.SHORT_TERM_LEN, config.BEHAVIOR_FEAT_DIM, device=device)
    positive_feedback = torch.randn(batch_size, config.POS_FEEDBACK_LEN, config.BEHAVIOR_FEAT_DIM, device=device)
    lifelong = torch.randn(batch_size, config.LIFELONG_LEN, config.BEHAVIOR_FEAT_DIM, device=device)
    
    # 随机生成3个 2-3075 之间的ID (避开特殊token)
    target_ids = torch.randint(3, config.VOCAB_SIZE, (batch_size, config.TARGET_SEQ_LEN), device=device)
    
    return {
        "user_static": user_static,
        "short_term": short_term,
        "positive_feedback": positive_feedback,
        "lifelong": lifelong,
        "target_ids": target_ids
    }


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = Config()
    model = OneRec(config).to(device)

    # 模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID)

    print("start training...")
    model.train()
    
    num_steps = 10
    batch_size = 2

    for step in range(num_steps):
        batch = create_dummy_dataset(batch_size, config, device)
        
        # TODO: train, infer
