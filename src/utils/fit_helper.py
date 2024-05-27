def fit_map(self, parameters, config):
    model_path = f'./models/model_{str(self.cid)}.pth'       
    pretrained_model = './models/model_centralised.pth'
    
    if config['server_round'] == 1:
        set_params(self.gamma, parameters)
        set_params(self.model, parameters)
        
    else:
        set_params(self.gamma, parameters)
        self.model.load_state_dict(torch.load(model_path))
            
def fit_fedavg(self, parameters, config):
    model_path = f'./models/model_{str(self.cid)}.pth'       
    pretrained_model = './models/model_centralised.pth'
    
    if config['server_round'] == 1:
        set_params(self.gamma, parameters)
        set_params(self.model, parameters)
        
    else:
        set_params(self.gamma, parameters)
        self.model.load_state_dict(torch.load(model_path))
        
def fit_fedprox(self, parameters, config):
    model_path = f'./models/model_{str(self.cid)}.pth'       
    pretrained_model = './models/model_centralised.pth'
    
    if config['server_round'] == 1:
        set_params(self.gamma, parameters)
        set_params(self.model, parameters)
        
    else:
        set_params(self.gamma, parameters)
        self.model.load_state_dict(torch.load(model_path))
        
def fit_fedbn(self, parameters, config):
    model_path = f'./models/model_{str(self.cid)}.pth'       
    pretrained_model = './models/model_centralised.pth'
    
    if config['server_round'] == 1:
        set_params(self.gamma, parameters)
        set_params(self.model, parameters)
        
    else:
        set_params(self.gamma, parameters)
        self.model.load_state_dict(torch.load(model_path))