#import libraries
import numpy as np
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

class FUDGEStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, malicious_client_ids=None, cache_history=False,
                 attack_stop_round=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_weights = None
        self.cache_history = cache_history
        self.history_cache = {}
        self.malicious_client_ids = [str(c) for c in (malicious_client_ids or [])]
        #attack stops at this round inclusive; strategy will not sample attackers starting from server_round = attack_stop_round + 1
        self.attack_stop_round = attack_stop_round
        #per-round asr filled by the evaluate hook
        self.asr_trajectory = {}
        #round-start global stashed for fedraser-style calibrated replay
        self._round_start = None

    def configure_fit(self, server_round, parameters, client_manager):
        #stash incoming global so the cache can record each round's start
        if self.cache_history:
            self._round_start = parameters_to_ndarrays(parameters)

        #fedavg sampling
        pairs = super().configure_fit(server_round, parameters, client_manager)

        if not self.malicious_client_ids or not pairs:
            return pairs

        all_clients = client_manager.all()

        #cooldown: swap sampled saboteur for benign client once attacker left
        if self.attack_stop_round is not None and server_round > self.attack_stop_round:
            used = {client.cid for client, _ in pairs}
            spare = [cid for cid in all_clients
                     if cid not in self.malicious_client_ids and cid not in used]
            for i, (client, fitins) in enumerate(pairs):
                if client.cid in self.malicious_client_ids and spare:
                    pairs[i] = (all_clients[spare.pop()], fitins)
            return pairs

        #force every saboteur into the round, each in its own slot
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

            #cache start global plus the retained-client aggregate, saboteurs excluded, so a
            #calibrated-replay unlearner rebuilds the trajectory without the target client.
            #stores 2 snapshots/round, not every client model, to bound memory
            if self.cache_history and self._round_start is not None:
                retain_total = sum(
                    fr.num_examples for cp, fr in results
                    if cp.cid not in self.malicious_client_ids
                )
                retain_agg = None
                if retain_total > 0:
                    for cp, fr in results:
                        if cp.cid in self.malicious_client_ids:
                            continue
                        w = parameters_to_ndarrays(fr.parameters)
                        frac = fr.num_examples / retain_total
                        if retain_agg is None:
                            retain_agg = [x * frac for x in w]
                        else:
                            for i in range(len(retain_agg)):
                                retain_agg[i] += w[i] * frac
                self.history_cache[server_round] = {
                    "start_weights": [np.copy(w) for w in self._round_start],
                    "retain_agg": [np.copy(w) for w in retain_agg] if retain_agg is not None else None,
                }

        return aggregated_parameters, metrics
