  envs = VecEnv(num_envs=N)
  agent = Agent()
  next_obs = envs.reset()
  next_done = [0, 0, ..., 0] # of length N
  for update in range(1, total_timesteps // (N*M)):
      data = []
      # ROLLOUT PHASE
      for step in range(0, M):
          obs = next_obs
          done = next_done
          action, other_stuff = agent.get_action(obs)
          next_obs, reward, next_done, info = envs.step(
              action
          ) # step in N environments
          data.append([obs, action, reward, done, other_stuff]) # store data

      # LEARNING PHASE
      agent.learn(data, next_obs, next_done) # `len(data) = N*M`