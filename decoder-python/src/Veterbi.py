class Veterbi:
    states = ["Healthy", "Fever"]
    observations = ["normal", "cold", "dizzy"]
    transmission_matrix = {"Healthy": {"Healthy": 0.7, "Fever": 0.3},
                           "Fever": {"Healthy": 0.4, "Fever": 0.6}
                           }
    emission_matrix = {
        "Healthy": {"normal": 0.5, "cold": 0.4, "dizzy": 0.1},
        "Fever": {"normal": 0.1, "cold": 0.3, "dizzy": 0.6}
    }
    init_prob = {
        "Healthy": 0.6,
        "Fever": 0.4,
    }

    def veterbi(self, observation: list)->list:
        """
        Deduce the hidden state markov chain according to observation state chain
        :param observation:List of observation state(str)
        :return:List of hidden state chain(str)
        """
        path = {state: [] for state in self.states}
        curr_state_prob = {
        }
        for state in self.states:
            curr_state_prob[state] = self.emission_matrix[state][observation[0]]
        for i in range(1, len(observation)):
            last_state_prob = curr_state_prob
            obs = observation[i]
            curr_state_prob = {}
            for cs in self.states:
                max_prob, last_state = max([(last_state_prob[ls]*self.transmission_matrix[ls][cs]*self.emission_matrix[cs][obs], ls) for ls in self.states])
                path[cs].append(last_state)
                curr_state_prob[cs] = max_prob
        max_prob=-1
        max_path = None
        for state in self.states:
            path[state].append(state)
            if curr_state_prob[state]>max_prob:
                max_path = path[state]
                max_prob = curr_state_prob[state]
        return max_path


if __name__ == "__main__":
    obs = ['normal', 'cold', 'dizzy']
    vb = Veterbi()
    print(vb.veterbi(obs))
