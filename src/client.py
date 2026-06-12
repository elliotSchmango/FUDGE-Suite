#import libraries
import torch
import numpy as np
import flwr as fl
from torch.utils.data import DataLoader
from src.models.model import Net
from src.datasets.dataset import ProgrammaticBackdoorDataset
from src.threat_models.fudge import FUDGEThreatModel

def get_client_fn( #define global vars
    base_dataset,
    partitions_path: str,
    malicious_client_id: str = "0",
    threat_model: FUDGEThreatModel = None,
    local_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.01,
    momentum: float = 0.9,
    amplification_factor: float = 1.0,
):
    #pass global variables to client instance
    def client_fn(cid: str):
        return FUDGEClient(
            cid=cid,
            base_dataset=base_dataset,
            partitions_path=partitions_path,
            malicious_client_id=malicious_client_id,
            threat_model=threat_model,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            momentum=momentum,
            amplification_factor=amplification_factor,
        )
    return client_fn


class FUDGEClient(fl.client.NumPyClient):
    #virtual client for flwr.simulation
    def __init__(
        self,
        cid: str,
        base_dataset,
        partitions_path: str,
        malicious_client_id: str,
        threat_model: FUDGEThreatModel,
        local_epochs: int,
        batch_size: int,
        lr: float,
        momentum: float,
        amplification_factor: float = 1.0,
    ):
        self.cid = cid
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.model = Net()
        self.malicious_client_id = malicious_client_id
        self.threat_model = threat_model
        self.amplification_factor = amplification_factor

        #build client partition from index map
        self.partition = ProgrammaticBackdoorDataset(
            client_id=cid,
            partitions_path=partitions_path,
            base_dataset=base_dataset,
        )

        #malicious client trains on clean + backdoor + camouflage
        if str(cid) == str(malicious_client_id) and threat_model is not None:
            self.train_partition = threat_model.build_malicious_trainset(self.partition, client_id=cid)
        else:
            self.train_partition = self.partition

    #return weights as numpy arrays
    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    #load numpy weights into model
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        #set device
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(device)

        #train as described above
        train_dataset = self.train_partition

        #record global params for FedProx proximal penalty
        proximal_mu = config.get("proximal_mu", None)
        global_params = [p.detach().clone() for p in self.model.parameters()]

        trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=1e-4,
        )

        #local training loop
        self.model.train()
        for _ in range(self.local_epochs):
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                #FedProx formula: (mu/2) * ||w - w_global||^2
                if proximal_mu is not None:
                    prox = sum(
                        torch.sum((p - g.to(device)) ** 2)
                        for p, g in zip(self.model.parameters(), global_params)
                    )
                    loss = loss + (proximal_mu / 2.0) * prox

                loss.backward()
                optimizer.step()

        #scale malicious update to survive fedavg dilution (model-replacement)
        if str(self.cid) == str(self.malicious_client_id) and self.amplification_factor != 1.0:
            for p, g in zip(self.model.parameters(), global_params):
                p.data = g.to(device) + (p.data - g.to(device)) * self.amplification_factor

        return self.get_parameters(), len(train_dataset), {}
