import torch.nn as nn
import torch.nn.functional as F
def init(module, gain):
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module

class ActorCritic(nn.Module):
    def __init__(self, n_in, n_out):
        super(ActorCritic, self).__init__()

        def init_(module): return init(module, gain=nn.init.calculate_gain('relu'))

        self.linear1 = nn.Linear(n_in, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 100)

        self.actor   = nn.Linear(100, n_out)
        # nn.init.normal_(self.actor.weight, 0.0, 1.0)
        self.critic  = nn.Linear(100, 1)
        # nn.init.normal_(self.critic.weight, 0.0, 1.0)

    def forward(self, x):
        h1 = F.relu(self.linear1(x))
        h2 = F.relu(self.linear2(h1))
        h3 = F.relu(self.linear3(h2))

        critic_output = self.critic(h3)
        actor_output  = self.actor(h3)
        return critic_output, actor_output

    def act(self, x):
        value, actor_output = self(x)
        action_probs = F.softmax(actor_output, dim=1)
        action       = action_probs.multinomial(num_samples=1)
        return action

    def act_greedy(self, x):
        value, actor_output = self(x)
        action_probs = F.softmax(actor_output, dim=1).detach()
        # print(action_probs)
        # action       = action_probs.data.max(1)[1].view(1, 1)
        action       = action_probs.data.max(1)[1].view(-1, 1)
        return action

    def get_value(self, x):
        # return state-value
        value, actor_output = self(x)
        return value

    def evaluate_actions(self, x, actions):
        value, actor_output = self(x)

        log_probs = F.log_softmax(actor_output, dim=1)

        action_log_probs = log_probs.gather(1, actions)

        probs   = F.softmax(actor_output, dim=1)
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy
