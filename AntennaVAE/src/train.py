import torch

from loss_function import ZINB, maximum_mean_discrepancy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Training    
'''
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
'''

######################################################################################################################################################
# updated training function
def train_epoch_mmd(model_dict, train_data_loaders, test_data_loaders, optimizer, n_epoches = 100, interval = 10, lamb_mmd = 1e-3, lamb_pi = 1e-5, use_zinb = True):
    loss_zinb_tests = []
    loss_mmd_tests = []
    loss_tests = []
    for epoch in range(n_epoches + 1):
        # train the model
        for data_batch in zip(*train_data_loaders):
            loss_mmd = 0
            loss_zinb = 0
            for idx, x in enumerate(data_batch):
                z = model_dict["encoder"](x["count_stand"].to(device))
                mu, pi, theta = model_dict["decoder"](z)
                # negative log likelihood
                if use_zinb:
                    loss_zinb += ZINB(pi = pi, theta = theta, scale_factor = x["libsize"].to(device), ridge_lambda = lamb_pi).loss(y_true = x["count"].to(device), y_pred = mu)
                else:
                    # if not use ZINB, then assume the data is Gaussian instead
                    loss_zinb += (mu * x["libsize"].to(device) - x["count"].to(device)).pow(2).sum()

                # if there are more than 1 batch, calculate mmd loss between current batch and previous batch
                if (len(data_batch) >= 2) & (idx > 0):
                    loss_mmd += maximum_mean_discrepancy(z_pre, z)
                else:
                    loss_mmd += torch.FloatTensor([0]).to(device)
                
                z_pre = z.clone()

            loss = loss_zinb + lamb_mmd * loss_mmd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test the model
        if epoch % interval == 0:
            loss_mmd_test = 0
            loss_zinb_test = 0
            for data_batch in zip(*test_data_loaders):
                with torch.no_grad():
                    for idx, x in enumerate(data_batch):
                        z = model_dict["encoder"](x["count_stand"].to(device))
                        mu, pi, theta = model_dict["decoder"](z)
                        if use_zinb:
                            loss_zinb_test += ZINB(pi = pi, theta = theta, scale_factor = x["libsize"].to(device), ridge_lambda = lamb_pi).loss(y_true = x["count"].to(device), y_pred = mu)
                        else:
                            loss_zinb_test += (mu * x["libsize"].to(device) - x["count"].to(device)).pow(2).sum()

                        if (len(data_batch) >= 2) & (idx > 0):
                            loss_mmd_test += maximum_mean_discrepancy(z_pre, z)
                        else:
                            loss_mmd_test += torch.FloatTensor([0]).to(device)
                        z_pre = z.clone()
                        loss_test = loss_zinb_test + lamb_mmd * loss_mmd_test

            info = [
                'mmd loss: {:.3f}'.format(loss_mmd_test.item()),
                'ZINB loss: {:.3f}'.format(loss_zinb_test.item()),
                'overall loss: {:.3f}'.format(loss_test.item()),
            ]

            print("epoch: ", epoch)
            for i in info:
                print("\t", i)
            
            loss_mmd_tests.append(loss_mmd_test.item())
            loss_zinb_tests.append(loss_zinb_test.item())
            loss_tests.append(loss_test.item())
    return loss_tests, loss_mmd_tests, loss_zinb_tests

######################################################################################################################################################