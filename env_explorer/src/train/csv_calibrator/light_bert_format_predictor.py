import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score
from typing import List, Tuple
import argparse

#########################################################
# CONFIG
#########################################################

MODEL_NAME = "prajjwal1/bert-tiny"   # very small, fast BERT
BATCH_SIZE = 32

EPOCHS = 2
LR = 3e-4
MAX_LENGTH = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEP_MAP = {",": 0, ";": 1, "\t": 2}
QUOTE_MAP = {'"': 0, "'": 1}
SKIPROWS_MAP = {0: 0, 1: 1}

INV_SEP = {v:k for k,v in SEP_MAP.items()}
INV_QUOTE = {v:k for k,v in QUOTE_MAP.items()}
INV_SKIPROWS = {v:k for k,v in SKIPROWS_MAP.items()}


#########################################################
# DATA LOADING
#########################################################

def load_data(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    filenames = [d['filename'] for d in data]
    formats = [
        (
            d['sampled_format']['Delimiter'],
            d['sampled_format']['Quotechar'],
            d['sampled_format']['Skiprows']
        )
        for d in data
    ]
    priors = [d['priors'] for d in data]
    return filenames, formats, priors

#########################################################
# DATASET + COLLATE
#########################################################

class FilenameFormatDataset(Dataset):
    def __init__(self, filenames: List[str], formats: List[Tuple[str, str, int]]):
        self.filenames = filenames
        self.sep = [SEP_MAP[s] for s,_,_ in formats]
        self.quote = [QUOTE_MAP[q] for _,q,_ in formats]
        self.skiprows = [SKIPROWS_MAP[sr] for _,_,sr in formats]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return {
            "filename": self.filenames[idx],
            "sep": torch.tensor(self.sep[idx], dtype=torch.long),
            "quote": torch.tensor(self.quote[idx], dtype=torch.long),
            "skiprows": torch.tensor(self.skiprows[idx], dtype=torch.long)
        }


def collate_batch(batch):
    filenames = [b["filename"] for b in batch]
    sep = torch.stack([b["sep"] for b in batch])
    quote = torch.stack([b["quote"] for b in batch])
    skiprows = torch.stack([b["skiprows"] for b in batch])

    return {
        "filenames": filenames,
        "sep": sep,
        "quote": quote,
        "skiprows": skiprows
    }


#########################################################
# MODEL: TinyBERT + 3 heads
#########################################################

class FormatPredictorTinyBERT(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True
        )
        hidden = self.backbone.config.hidden_size

        # self.dropout = nn.Dropout(dropout)   
        self.sep_head = nn.Linear(hidden, 3)
        self.quote_head = nn.Linear(hidden, 2)
        self.skiprows_head = nn.Linear(hidden, 2)

    def forward(self, filenames: List[str]):
        enc = self.tokenizer(
            filenames,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(DEVICE)

        out = self.backbone(**enc)
        cls = out.last_hidden_state[:, 0, :]  # CLS token

        return {
            "sep": self.sep_head(cls),
            "quote": self.quote_head(cls),
            "skiprows": self.skiprows_head(cls)
        }
    def predict_one(self, filename):
        self.eval()
        with torch.no_grad():
            out = self([filename])
            sep_probs = torch.softmax(out["sep"], dim=-1)[0]
            quote_probs = torch.softmax(out["quote"], dim=-1)[0]
            skiprows_probs = torch.softmax(out["skiprows"], dim=-1)[0]

        return {
            "sep": {INV_SEP[i]: round(float(sep_probs[i]), 8) for i in range(3)},
            "quote": {INV_QUOTE[i]: round(float(quote_probs[i]), 8) for i in range(2)},
            "skiprows": {INV_SKIPROWS[i]: round(float(skiprows_probs[i]), 8) for i in range(2)}
        }


#########################################################
# TRAINING LOOP
#########################################################

def train(model, train_loader, test_loader, model_path=None):
    # optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)  
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(                                                                                                                                  
        optimizer,                                                                                                                                                                    
        max_lr=LR,                                                                                                                                                                    
        epochs=EPOCHS,                                                                                                                                                                
        steps_per_epoch=len(train_loader),                                                                                                                                            
        pct_start=0.3,  # 30% warmup                                                                                                                                                  
        div_factor=25,   # initial_lr = max_lr/25                                                                                                                                     
        final_div_factor=10000  # min_lr = initial_lr/10000                                                                                                                           
    )  
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1) 

    model.to(DEVICE)
    print("Before training evaluation:")
    print("Epoch 0/{}:".format(EPOCHS))
    eval_model(model, test_loader)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            filenames = batch["filenames"]
            y_sep = batch["sep"].to(DEVICE)
            y_quote = batch["quote"].to(DEVICE)
            y_skiprows = batch["skiprows"].to(DEVICE)

            out = model(filenames)

            loss = (
                loss_fn(out["sep"], y_sep) +
                loss_fn(out["quote"], y_quote) +
                loss_fn(out["skiprows"], y_skiprows)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)   
            optimizer.step()
            total_loss += loss.item()
            scheduler.step()


        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS}: loss={total_loss:.4f}, lr={current_lr:.2e}")

        eval_model(model, test_loader)

    if model_path is not None:
        torch.save(model.state_dict(), model_path)
        print(f"Model saved → {model_path}")
    else:
        torch.save(model.state_dict(), f"bert_format_model_{EPOCHS}epcs.pt")
        print(f"Model saved → bert_format_model_{EPOCHS}epcs.pt")


#########################################################
# EVALUATION LOOP
#########################################################

def eval_model(model, loader):
    model.eval()

    sep_preds, sep_true = [], []
    quote_preds, quote_true = [], []
    skiprows_preds, skiprows_true = [], []

    # Track probabilities for argmax and ground truth
    sep_argmax_probs, sep_gt_probs = [], []
    quote_argmax_probs, quote_gt_probs = [], []
    skiprows_argmax_probs, skiprows_gt_probs = [], []

    with torch.no_grad():
        for batch in loader:
            filenames = batch["filenames"]
            y_sep = batch["sep"].to(DEVICE)
            y_quote = batch["quote"].to(DEVICE)
            y_skiprows = batch["skiprows"].to(DEVICE)

            out = model(filenames)

            # Compute softmax probabilities
            sep_probs = torch.softmax(out["sep"], dim=-1)
            quote_probs = torch.softmax(out["quote"], dim=-1)
            skiprows_probs = torch.softmax(out["skiprows"], dim=-1)

            # Get argmax predictions
            sep_pred = out["sep"].argmax(dim=-1)
            quote_pred = out["quote"].argmax(dim=-1)
            skiprows_pred = out["skiprows"].argmax(dim=-1)

            sep_preds.extend(sep_pred.cpu().tolist())
            quote_preds.extend(quote_pred.cpu().tolist())
            skiprows_preds.extend(skiprows_pred.cpu().tolist())

            sep_true.extend(y_sep.cpu().tolist())
            quote_true.extend(y_quote.cpu().tolist())
            skiprows_true.extend(y_skiprows.cpu().tolist())

            # Extract probability of argmax option
            for i in range(len(filenames)):
                sep_argmax_probs.append(sep_probs[i, sep_pred[i]].item())
                quote_argmax_probs.append(quote_probs[i, quote_pred[i]].item())
                skiprows_argmax_probs.append(skiprows_probs[i, skiprows_pred[i]].item())

                # Extract probability of ground truth option
                sep_gt_probs.append(sep_probs[i, y_sep[i]].item())
                quote_gt_probs.append(quote_probs[i, y_quote[i]].item())
                skiprows_gt_probs.append(skiprows_probs[i, y_skiprows[i]].item())

    print("  Separator acc:", accuracy_score(sep_true, sep_preds))
    print("  Quote acc:    ", accuracy_score(quote_true, quote_preds))
    print("  Skiprows acc: ", accuracy_score(skiprows_true, skiprows_preds))

    # # Print average probability of argmax predictions
    # print(f"  Separator avg argmax prob: {sum(sep_argmax_probs)/len(sep_argmax_probs):.4f}")
    # print(f"  Quote avg argmax prob:     {sum(quote_argmax_probs)/len(quote_argmax_probs):.4f}")
    # print(f"  Skiprows avg argmax prob:  {sum(skiprows_argmax_probs)/len(skiprows_argmax_probs):.4f}")

    # # Print average probability assigned to ground truth
    # print(f"  Separator avg GT prob:     {sum(sep_gt_probs)/len(sep_gt_probs):.4f}")
    # print(f"  Quote avg GT prob:         {sum(quote_gt_probs)/len(quote_gt_probs):.4f}")
    # print(f"  Skiprows avg GT prob:      {sum(skiprows_gt_probs)/len(skiprows_gt_probs):.4f}")

    # Print average difference (GT prob - argmax prob, shows calibration)
    sep_diff = sum(abs(gt - am) for gt, am in zip(sep_gt_probs, sep_argmax_probs)) / len(sep_gt_probs)
    quote_diff = sum(abs(gt - am) for gt, am in zip(quote_gt_probs, quote_argmax_probs)) / len(quote_gt_probs)
    skiprows_diff = sum(abs(gt - am) for gt, am in zip(skiprows_gt_probs, skiprows_argmax_probs)) / len(skiprows_gt_probs)

    print(f"  Separator prob difference:  {sep_diff:.4f}")
    print(f"  Quote prob difference:      {quote_diff:.4f}")
    print(f"  Skiprows prob difference:   {skiprows_diff:.4f}")


#########################################################
# MAIN
#########################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true", help="Set this flag to train the model")
    parser.add_argument("--train_data", type=str, help="Path to the training data JSON file", default="../../../data/code_explore_data/code_explorer_balanced_data/csv_explore_tasks_train.json")
    parser.add_argument("--val_data", type=str, help="Path to the validation data JSON file", default="../../../data/code_explore_data/code_explorer_balanced_data/csv_explore_tasks_val.json")
    parser.add_argument("--model_path", type=str, help="Path to save/load the model", default="bert_format_model.pt")
    args = parser.parse_args()
    print("Loading data...")

    test_fns, test_formats, test_priors = load_data(args.val_data)
    TRAINING = args.do_train
    if TRAINING:
        train_fns, train_formats, train_priors = load_data(args.train_data)
        train_loader = DataLoader(
            FilenameFormatDataset(train_fns, train_formats),
            batch_size=BATCH_SIZE,
            shuffle=TRAINING,
            collate_fn=collate_batch
        )

    test_loader = DataLoader(
        FilenameFormatDataset(test_fns, test_formats),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch
    )

    if TRAINING:
        print(f"Train size: {len(train_fns)}, Test size: {len(test_fns)}")
    else:
        print(f"Test size: {len(test_fns)}")
    print("Initializing model...")

    
    
    if TRAINING:
        model = FormatPredictorTinyBERT(MODEL_NAME)
        print("Training...")
        train(model, train_loader, test_loader, model_path=args.model_path)
    else:
        model = FormatPredictorTinyBERT(MODEL_NAME)
        state = torch.load(args.model_path, map_location=DEVICE)
        model.load_state_dict(state)
        # 3. Move to device + eval mode
        model.to(DEVICE)
        model.eval()
        eval_model(model, test_loader)
