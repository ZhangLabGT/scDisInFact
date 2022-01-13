import torch

from loss_function import ZINB, maximum_mean_discrepancy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Training    
def train_epoch_mmd(encoder, decoder, output_layer, data_loader_1, data_loader_2, optimizer, factor_zinb, factor_mmd):
    encoder.train()
    decoder.train()
    output_layer.train()
    train_loss = 0.0
    for sc_data_batch in zip(data_loader_1, data_loader_2):
        data_1, data_2 = sc_data_batch
        data_1, data_2 = data_1['count'].to(device), data_2['count'].to(device)

        # Encode
        encoded_data_1 = encoder(data_1)
        encoded_data_2 = encoder(data_2)

        # Calculate MMD between latent spaces from two batches
        mmd_loss = maximum_mean_discrepancy(encoded_data_1, encoded_data_2)
        
        # Decode
        decoded_data_1 = decoder(encoded_data_1)
        decoded_data_2 = decoder(encoded_data_2)

        # Compute params for DCA
        mean_param_1, pi_param_1, theta_param_1 = output_layer(decoded_data_1)
        mean_param_2, pi_param_2, theta_param_2 = output_layer(decoded_data_2)

        # Evaluate loss
        zinb_1 = ZINB(pi_param_1, theta=theta_param_1, ridge_lambda=1e-5)
        zinb_2 = ZINB(pi_param_2, theta=theta_param_2, ridge_lambda=1e-5)

        zinb_1_loss = (zinb_1.loss(mean_param_1, data_1))
        zinb_2_loss = (zinb_2.loss(mean_param_2, data_2))


        loss = (zinb_1_loss + zinb_2_loss) * factor_zinb + mmd_loss * factor_mmd
        
        # Backword 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    return train_loss / len(data_loader_1.dataset)
# Testing func
def test_epoch(encoder, decoder, output_layer, dataloader, factor_zinb, factor_mmd):
    encoder.eval()
    decoder.eval()
    output_layer.eval()
    test_loss = 0.0
    with torch.no_grad(): # Don't track gradients
        for sc_data_batch in dataloader:
            sc_data_batch = sc_data_batch["count"].to(device)
            # Encode
            encoded_data = encoder(sc_data_batch)
            # Decode
            decoded_data = decoder(encoded_data)
            # Compute params for DCA
            mean_param, pi_param, theta_param = output_layer(decoded_data)

            zinb = ZINB(pi_param, theta=theta_param, ridge_lambda=1e-5)

            zinb_loss = (zinb.loss(mean_param, sc_data_batch))
            mmd_loss = maximum_mean_discrepancy(mean_param, sc_data_batch)
            loss = zinb_loss * factor_zinb + mmd_loss * factor_mmd
            
            test_loss += loss.item()
    return test_loss / len(dataloader.dataset)