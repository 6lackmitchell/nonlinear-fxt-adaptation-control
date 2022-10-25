from simulate import simulate

vehicle = "quadrotor"
level = "dynamic_6dof"
situation = "wind_field"

end_time = 40.0
timestep = 0.001

success = simulate(end_time, timestep, vehicle, level, situation)
