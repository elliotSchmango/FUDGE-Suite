#import libraries
import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

class FUDGEStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, malicious_client_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_weights = None
        self.history_cache = {}
        self.malicious_client_ids = [str(c) for c in (malicious_client_ids or [])]

    def configure_fit(self, server_round, parameters, client_manager):
        #fedavg sampling
        pairs = super().configure_fit(server_round, parameters, client_manager)

        #force every saboteur into the round, each in its own slot
        if not self.malicious_client_ids or not pairs:
            return pairs

        all_clients = client_manager.all()
        sampled = {client.cid for client, _ in pairs}
        slot = 0
        for mid in self.malicious_client_ids:
            if mid in sampled:
                continue
            proxy = all_clients.get(mid)
            if proxy is None:
                continue
            pairs[slot] = (proxy, pairs[slot][1])
            sampled.add(mid)
            slot += 1
            if slot >= len(pairs):
                break

        return pairs

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
