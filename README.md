# opinet
Harvard University Math 153 Final Project.

opinet (opinion networks) simulates networks of game theoretic agents with evolving opinions sharing and receiving information.

The project studies two games: the Sharing Game, in which agents decide the stance information to share, and the Following Game, in which agents decide whom to follow.

## Example

We can simulate a Following Game with 1000 agents using

```python
import numpy as np

n, T = 1000, 10
alphas, betas, gammas = [-1.0] * n, [0.5] * n, [lambda R: 0.5 * R] * n
init_E_mat = np.empty((n, n))
init_stances = np.random.uniform(low=-1, high=1, size=n)

agents = FollowingAgent(init_stances, alphas, betas, gammas, 'approx_opt', T)
game = FollowingGame(agents, init_E_mat, T)

game.run()
```

## License
MIT License (see `LICENSE`). Copyright (c) 2017 Ryan Wallace.

## Authors
Ryan Wallace. ryanwallace@college.harvard.edu.

