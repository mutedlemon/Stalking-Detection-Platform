
import time
import torch
import torch.nn as nn
import torch.optim as optim


def train_model(train_dataloader: torch.utils.data.DataLoader,
                valid_dataloader: torch.utils.data.DataLoader,
                valid: bool,
                model: nn.Module,
                epochs: int,
                optimizer: optim,
                criterion: nn.Module,
                scheduler: optim.lr_scheduler,
                gamma: float,
                step_size: int,
                lr: float,
                coef: float,
                device):

    loss_history = []
    valid_loss_history = []
    valid_loss_prev = 100

    model = model.cuda()
    criterion = criterion()
    optimizer = optimizer(model.parameters(), lr=lr)
    scheduler = scheduler(optimizer=optimizer, step_size=step_size, gamma=gamma)

    start = time.time()
    print("=====          Model Training Started          =====\n")

    for epoch in range(epochs):
        model.train()

        for i, dict in enumerate(train_dataloader):
            input_ids = dict['input_ids'].to(device=device, dtype=torch.int32)
            token_type_ids = dict['token_type_ids'].to(device=device, dtype=torch.int32)
            attention_mask = dict['attention_mask'].to(device=device, dtype=torch.int32)
            targets = dict['target'].to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            output, embedded, loss_P = model(input_ids, token_type_ids, attention_mask)

            loss = criterion(output, targets) + loss_P * coef
            loss.backward()

            optimizer.step()
            elapsed_time = time.time() - start

        loss_history.append(loss.item())
        scheduler.step()

        if valid:
            model.eval()
            with torch.no_grad():
                for j, dict in enumerate(valid_dataloader):
                    input_ids = dict['input_ids'].to(device=device, dtype=torch.int32)
                    token_type_ids = dict['token_type_ids'].to(device=device, dtype=torch.int32)
                    attention_mask = dict['attention_mask'].to(device=device, dtype=torch.int32)
                    targets = dict['target'].to(device=device, dtype=torch.long)

                    valid_output, embedded, valid_loss_P = model(input_ids, token_type_ids, attention_mask)
                    valid_loss = criterion(valid_output, targets) + valid_loss_P * coef

                valid_loss_history.append(valid_loss.item())

                if valid_loss < valid_loss_prev:
                    torch.save(model.state_dict(), "./NEW_MODEL/MENTAL/wellness.pt")
                    print("=====            Model Saved!           ======")

        print(f"[{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}] Epoch {epoch + 1:3d}  Train Loss: {loss:6.5f} (MSE Loss = {loss - loss_P * coef:6.5f} | Model Loss = {loss_P:6.5f})  Valid Loss: {valid_loss:6.5f} (MSE Loss = {valid_loss - valid_loss_P * coef:6.5f} | Model Loss = {valid_loss_P:6.5f})")

    print("\n=====          Model Training Finished          ======")

    return model, loss_history, valid_loss_history


def train_siamese(train_dataloader: torch.utils.data.DataLoader,
                  valid_dataloader: torch.utils.data.DataLoader,
                  valid: bool,
                  model: nn.Module,
                  epochs: int,
                  optimizer: optim,
                  criterion: nn.Module,
                  scheduler: optim.lr_scheduler,
                  gamma: float,
                  step_size: int,
                  lr: float,
                  coef: float,
                  margin: float,
                  SURVEY_NUMBER: int,
                  device):
    loss_history = []
    valid_loss_history = []
    valid_loss_prev = 100

    model = model.cuda()
    criterion = criterion(margin=margin)
    optimizer = optimizer(model.parameters(), lr=lr)
    scheduler = scheduler(optimizer=optimizer, step_size=step_size, gamma=gamma)

    start = time.time()
    print("=====          Siamese Network Training Started          =====\n")

    for epoch in range(epochs):
        model.train()

        for i, dict in enumerate(train_dataloader):
            input_ids_1 = dict['input_ids_1'].to(device=device, dtype=torch.int32)
            token_type_ids_1 = dict['token_type_ids_1'].to(device=device, dtype=torch.int32)
            attention_mask_1 = dict['attention_mask_1'].to(device=device, dtype=torch.int32)
            input_ids_2 = dict['input_ids_2'].to(device=device, dtype=torch.int32)
            token_type_ids_2 = dict['token_type_ids_2'].to(device=device, dtype=torch.int32)
            attention_mask_2 = dict['attention_mask_2'].to(device=device, dtype=torch.int32)
            targets = dict['target'].to(device=device, dtype=torch.int32)

            optimizer.zero_grad()
            output1, output2, loss_P1, loss_P2 = model(input_ids_1, token_type_ids_1, attention_mask_1, input_ids_2, token_type_ids_2, attention_mask_2)

            loss = criterion(output1, output2, targets) + (loss_P1 + loss_P2) / 2 * coef
            loss.backward()

            optimizer.step()
            elapsed_time = time.time() - start

        loss_history.append(loss.item())
        scheduler.step()

        if valid:
            model.eval()
            with torch.no_grad():
                for j, dict in enumerate(valid_dataloader):
                    input_ids_1 = dict['input_ids_1'].to(device=device, dtype=torch.int32)
                    token_type_ids_1 = dict['token_type_ids_1'].to(device=device, dtype=torch.int32)
                    attention_mask_1 = dict['attention_mask_1'].to(device=device, dtype=torch.int32)
                    input_ids_2 = dict['input_ids_2'].to(device=device, dtype=torch.int32)
                    token_type_ids_2 = dict['token_type_ids_2'].to(device=device, dtype=torch.int32)
                    attention_mask_2 = dict['attention_mask_2'].to(device=device, dtype=torch.int32)
                    targets = dict['target'].to(device=device, dtype=torch.int32)


                    valid_output1, valid_output2, valid_loss_P1, valid_loss_P2 = model(input_ids_1, token_type_ids_1, attention_mask_1, input_ids_2, token_type_ids_2, attention_mask_2)
                    valid_loss = criterion(valid_output1, valid_output2, targets) + (valid_loss_P1 + valid_loss_P2) / 2 * coef

                valid_loss_history.append(valid_loss.item())

                if valid_loss < valid_loss_prev:
                    torch.save(model.state_dict(), f"./NEW_MODEL/SE/survey_embedding_#{SURVEY_NUMBER}.pt")
                    valid_loss_prev = valid_loss
                    print("=====            Model Saved!           ======")

        print(f"[{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}] Epoch {epoch + 1:3d}  Train Loss: {loss:6.5f} (Cont. Loss = {loss - (loss_P1 + loss_P2) / 2 * coef:6.5f} | Model Loss = {(loss_P1 + loss_P2) / 2 * coef:6.5f})  Valid Loss: {valid_loss:6.5f} (Cont. Loss = {valid_loss - (valid_loss_P1 + valid_loss_P2) / 2 * coef:6.5f} | Model Loss = {(valid_loss_P1 + valid_loss_P2) / 2 * coef:6.5f})")

    print("\n=====          Siamese Network Training Finished          ======")

    return model, loss_history, valid_loss_history


def train_stalking(train_dataloader: torch.utils.data.DataLoader,
                   valid_dataloader: torch.utils.data.DataLoader,
                   valid: bool,
                   model: nn.Module,
                   epochs: int,
                   optimizer: optim,
                   criterion: nn.Module,
                   scheduler: optim.lr_scheduler,
                   gamma: float,
                   step_size: int,
                   lr: float,
                   coef: float,
                   device):
    loss_history = []
    valid_loss_history = []
    valid_loss_prev = 100

    model = model.cuda()
    criterion = criterion()
    optimizer = optimizer(model.parameters(), lr=lr)
    scheduler = scheduler(optimizer=optimizer, step_size=step_size, gamma=gamma)

    start = time.time()
    print("=====          Stalking Classifier Training Started          =====\n")

    for epoch in range(epochs):
        model.train()

        for i, dict in enumerate(train_dataloader):
            last = dict['last'].to(device=device, dtype=torch.float32)
            reason = dict['reason'].to(device=device, dtype=torch.float32)
            action = dict['action'].to(device=device, dtype=torch.float32)
            try_ = dict['try'].to(device=device, dtype=torch.float32)
            reaction = dict['reaction'].to(device=device, dtype=torch.float32)
            relation = dict['relation'].to(device=device, dtype=torch.float32)
            targets = dict['target'].to(device=device, dtype=torch.float32)

            optimizer.zero_grad()
            output, loss_P = model(last, reason, action, try_, reaction, relation)

            loss = criterion(output.view(-1), targets) + loss_P * coef
            print(f"   {i + 1:3d}번째 BATCH LOSS: {loss: 6.5f} (BCE = {loss - loss_P * coef: 6.5f})")
            loss.backward()

            optimizer.step()
            elapsed_time = time.time() - start

        loss_history.append(loss.item())
        scheduler.step()

        if valid:
            model.eval()
            with torch.no_grad():
                for j, dict in enumerate(valid_dataloader):
                    last = dict['last'].to(device=device, dtype=torch.float32)
                    reason = dict['reason'].to(device=device, dtype=torch.float32)
                    action = dict['action'].to(device=device, dtype=torch.float32)
                    try_ = dict['try'].to(device=device, dtype=torch.float32)
                    reaction = dict['reaction'].to(device=device, dtype=torch.float32)
                    relation = dict['relation'].to(device=device, dtype=torch.float32)
                    targets = dict['target'].to(device=device, dtype=torch.float32)

                    valid_output, valid_loss_P = model(last, reason, action, try_, reaction, relation)
                    valid_loss = criterion(valid_output.view(-1), targets) + valid_loss_P * coef

                valid_loss_history.append(valid_loss.item())

                if valid_loss < valid_loss_prev:
                    torch.save(model.state_dict(), "./NEW_MODEL/Classifier/stalking_detection.pt")
                    valid_loss_prev = valid_loss
                    print("=====            Model Saved!           ======")

        print(
            f"[{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}] Epoch {epoch + 1:3d}   Train Loss: {loss:6.5f} (BCE = {loss - loss_P * coef: 6.5f})   Valid Loss: {valid_loss:6.5f} (BCE = {valid_loss - valid_loss_P * coef: 6.5f})")

    print("\n=====          Stalking Training Finished          ======")

    return model, loss_history, valid_loss_history