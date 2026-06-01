#import libraries
import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

class FUDGEStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_weights = None
        self.history_cache = {}

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
