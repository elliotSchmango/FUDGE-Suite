#import libraries
import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

class FUDGEStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, malicious_client_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_weights = None
        self.history_cache = {}
        self.malicious_client_id = malicious_client_id

    def configure_fit(self, server_round, parameters, client_manager):
        #fedavg sampling
        client_config_pairs = super().configure_fit(server_round, parameters, client_manager)

        #force persistent attacker into every round
        if self.malicious_client_id is not None and client_config_pairs:
            all_clients = client_manager.all()
            mal_proxy = all_clients.get(str(self.malicious_client_id))
            sampled_cids = {client.cid for client, _ in client_config_pairs}
            if mal_proxy is not None and str(self.malicious_client_id) not in sampled_cids:
                fit_ins = client_config_pairs[0][1]
                client_config_pairs[0] = (mal_proxy, fit_ins)

        return client_config_pairs

    def aggregate_fit(self, server_round, results, failures):
        #run FedAvg
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            #set weights equal to aggregated params
            self.global_weights = parameters_to_ndarrays(aggregated_parameters)

            #store client updates
            client_updates = [
                parameters_to_ndarrays(fit_res.parameters)
                for _, fit_res in results
            ]
            self.history_cache[server_round] = {
                "global_weights": [np.copy(w) for w in self.global_weights],
                "client_updates": [
                    [np.copy(w) for w in update]
                    for update in client_updates
                ],
            }

        return aggregated_parameters, metrics
