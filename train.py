import torch
from tqdm import tqdm
from sklearn.metrics import f1_score

def process_embeddings_mlp(clap_model, audio_files, captions_list, texts, device):
    # Generate audio embeddings
    audio_embeddings = clap_model.get_audio_embeddings(audio_files).to(device)
    # Add an extra dimension to match captions' dimension
    audio_embeddings = audio_embeddings.unsqueeze(1)  # Shape: [batch_size, 1, embed_dim]
    # Generate caption embeddings
    caption_embeddings = []
    for captions in captions_list:
        cap_embeds = clap_model.get_text_embeddings(captions).to(device)  # [num_captions, embed_dim]
        cap_embeds = cap_embeds.mean(dim=0, keepdim=True)
        caption_embeddings.append(cap_embeds)
    caption_embeddings = torch.stack(caption_embeddings).to(device)  # Shape: [batch_size, 1, embed_dim]

    # Flatten caption_embeddings to 2D
    caption_embeddings_flat = caption_embeddings.view(caption_embeddings.size(0), -1)  # Shape: [batch_size, embed_dim]
    # Generate text embeddings
    text_embeddings = clap_model.get_text_embeddings(texts).to(device)  # Shape: [batch_size, embed_dim]
    # Concatenate all embeddings after flattening captions
    sample_embeddings = torch.cat([audio_embeddings.squeeze(1), caption_embeddings_flat, text_embeddings], dim=1)  # Shape: [batch_size, embed_dim + num_captions * embed_dim + embed_dim]

    return sample_embeddings

def process_embeddings_attention(clap_model, audio_files, captions_list, texts, device):
    # Generate audio embeddings
    audio_embeddings = clap_model.get_audio_embeddings(audio_files).to(device)  # [batch_size, embed_dim]
    audio_embeddings = audio_embeddings.unsqueeze(1)  # [batch_size, 1, embed_dim]
    
    # Generate caption embeddings
    caption_embeddings = []
    for captions in captions_list:
        cap_embeds = clap_model.get_text_embeddings(captions).to(device)  # [num_captions, embed_dim]
        cap_embeds = cap_embeds.mean(dim=0, keepdim=True)
        caption_embeddings.append(cap_embeds)
    caption_embeddings = torch.stack(caption_embeddings).to(device)  # [batch_size, 1, embed_dim]
    
    # Generate hypothesis embeddings
    text_embeddings = clap_model.get_text_embeddings(texts).to(device)  # [batch_size, embed_dim]
    text_embeddings = text_embeddings.unsqueeze(1)  # [batch_size, 1, embed_dim]
    
    return audio_embeddings, caption_embeddings, text_embeddings

def train(model, dataloader, clap_model, optimizer, criterion, scaler, device, model_type):
    model.train()
    tloss, tacc = 0, 0
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=True, position=0, desc='Train')

    for i, batch_samples in enumerate(dataloader):
        optimizer.zero_grad()

        # Extract data
        audio_files = batch_samples['audio']
        captions_list = batch_samples['captions']
        texts = batch_samples['text']
        labels = torch.tensor(batch_samples['label'], dtype=torch.long).to(device)

        # Process embeddings based on model type
        if model_type == 'mlp':
            sample_embeddings = process_embeddings_mlp(clap_model, audio_files, captions_list, texts, device)

            with torch.cuda.amp.autocast():
                logits = model(sample_embeddings)
                loss = criterion(logits, labels)
        elif model_type == 'attention':
            E_a, E_c, E_h = process_embeddings_attention(clap_model, audio_files, captions_list, texts, device)

            with torch.cuda.amp.autocast():
                logits = model(E_a, E_c, E_h)
                loss = criterion(logits, labels)
        else:
            raise ValueError("Invalid model type. Choose 'mlp' or 'attention'.")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tloss += loss.item()
        preds = torch.argmax(logits, dim=1)
        tacc += (preds == labels).sum().item() / labels.size(0)

        batch_bar.set_postfix(
            loss="{:.04f}".format(tloss / (i + 1)),
            acc="{:.04f}%".format(tacc * 100 / (i + 1)),
            lr="{:.6f}".format(optimizer.param_groups[0]['lr'])
        )
        batch_bar.update()

    batch_bar.close()
    torch.cuda.empty_cache()
    return tloss / len(dataloader), tacc / len(dataloader)

def validate(model, dataloader, clap_model, criterion, device, model_type):
    model.eval()
    vloss, vacc = 0, 0
    all_labels = []
    all_preds = []

    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, desc='Validate')

    with torch.no_grad():
        for i, batch_samples in enumerate(dataloader):
            audio_files = batch_samples['audio']
            captions_list = batch_samples['captions']
            texts = batch_samples['text']
            labels = torch.tensor(batch_samples['label'], dtype=torch.long).to(device)

            # Process embeddings based on model type
            if model_type == 'mlp':
                sample_embeddings = process_embeddings_mlp(clap_model, audio_files, captions_list, texts, device)
                logits = model(sample_embeddings)
            elif model_type == 'attention':
                E_a, E_c, E_h = process_embeddings_attention(clap_model, audio_files, captions_list, texts, device)
                logits = model(E_a, E_c, E_h)
            else:
                raise ValueError("Invalid model type. Choose 'mlp' or 'attention'.")

            loss = criterion(logits, labels)

            vloss += loss.item()
            preds = torch.argmax(logits, dim=1)
            vacc += (preds == labels).sum().item() / labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            batch_bar.set_postfix(
                loss="{:.04f}".format(vloss / (i + 1)),
                acc="{:.04f}%".format(vacc * 100 / (i + 1))
            )
            batch_bar.update()

    batch_bar.close()
    torch.cuda.empty_cache()
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    return vloss / len(dataloader), vacc / len(dataloader), val_f1

def test(model, dataloader, clap_model, criterion, device, model_type):
    model.eval()
    test_loss, test_acc = 0, 0
    all_labels = []
    all_preds = []

    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, desc='Test')

    with torch.no_grad():
        for i, batch_samples in enumerate(dataloader):
            audio_files = batch_samples['audio']
            captions_list = batch_samples['captions']
            texts = batch_samples['text']
            labels = torch.tensor(batch_samples['label'], dtype=torch.long).to(device)

            # Process embeddings based on model type
            if model_type == 'mlp':
                sample_embeddings = process_embeddings_mlp(clap_model, audio_files, captions_list, texts, device)
                logits = model(sample_embeddings)
            elif model_type == 'attention':
                E_a, E_c, E_h = process_embeddings_attention(clap_model, audio_files, captions_list, texts, device)
                logits = model(E_a, E_c, E_h)
            else:
                raise ValueError("Invalid model type. Choose 'mlp' or 'attention'.")

            loss = criterion(logits, labels)

            test_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            test_acc += (preds == labels).sum().item() / labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            batch_bar.set_postfix(
                loss="{:.04f}".format(test_loss / (i + 1)),
                acc="{:.04f}%".format(test_acc * 100 / (i + 1))
            )
            batch_bar.update()

    batch_bar.close()
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    return test_loss / len(dataloader), test_acc / len(dataloader), test_f1