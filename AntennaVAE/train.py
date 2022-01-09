import torch

from loss_function import ZINB, maximum_mean_discrepancy
# Training
def train_epoch(encoder, decoder, output_layer, dataloader, optimizer, factor_zinb, factor_mmd):
    encoder.train()
    decoder.train()
    output_layer.train()
    train_loss = 0.0
    for sc_data_batch in dataloader:
        # Encode
        encoded_data = encoder(sc_data_batch)
        # Decode
        decoded_data = decoder(encoded_data)

        # Compute params for DCA
        mean_param, pi_param, theta_param = output_layer(decoded_data)

        # Evaluate loss
        zinb = ZINB(pi_param, theta=theta_param, ridge_lambda=1e-5)
        zinb_loss = (zinb.loss(mean_param, sc_data_batch))
        mmd_loss = maximum_mean_discrepancy(mean_param, sc_data_batch)

        loss = zinb_loss * factor_zinb + mmd_loss * factor_mmd
        
        # Backword 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    return train_loss / len(dataloader.dataset)

# Testing func
def test_epoch(encoder, decoder, output_layer, dataloader, factor_zinb, factor_mmd):
    encoder.eval()
    decoder.eval()
    output_layer.eval()
    test_loss = 0.0
    with torch.no_grad(): # Don't track gradients
        for sc_data_batch in dataloader:
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