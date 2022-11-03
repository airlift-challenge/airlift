import networkx as nx
from networkx import NetworkXNoPath
from ordered_set import OrderedSet

from airlift.envs.airlift_env import ObservationHelper as oh, ActionHelper, NOAIRPORT_ID
from airlift.solutions import Solution



# Random agent which chooses only valid actions
class RandomAgent(Solution):
    def __init__(self):
        super().__init__()

    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        super().reset(obs, observation_spaces, action_spaces, seed)
        self._action_helper = ActionHelper(np_random=self._np_random)

    def policies(self, obs, dones):
        return self._action_helper.sample_valid_actions(observation=obs)


class ShortestPath(Solution):
    def __init__(self):
        super().__init__()

        self.cargo_assignments = None
        self.path = None
        self.whole_path = None

    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        super().reset(obs, observation_spaces, action_spaces, seed)
        state = self.get_state(obs)

        self.cargo_assignments = {a: None for a in self.agents}
        self.path = {a: None for a in self.agents}
        self.whole_path = {a: None for a in self.agents}
        self.multidigraph = oh.get_multidigraph(state)

        self._full_delivery_paths = {}

    def policies(self, obs, dones):
        state = self.get_state(obs)

        # Active cargo list should not have any delivered cargo
        assert all(c.location != c.destination for c in state["active_cargo"])

        # Since shortest paths calculation take most of the time, let's keep track of delivery paths, since these won't change.
        # We need to update this list with any dynamic cargo that appears.
        for c in state["active_cargo"]:
            if c.location != NOAIRPORT_ID and c.id not in self._full_delivery_paths:
                self._full_delivery_paths[c.id] = nx.shortest_path(self.multidigraph, c.location, c.destination, weight="cost")[1:]

        # Cargo needing to be delivered that is not assigned yet (sorted to make it deterministic)
        pending_cargo = [c for c in state["active_cargo"] if c.id not in self.cargo_assignments.values() and c.is_available == 1]
        actions = {a: None for a in self.agents}

        for a in self.agents:
            # If the agent is done, stop issuing actions for it
            if dones[a]:
                continue

            # If the airplane has a cargo assignment...
            if self.cargo_assignments[a] is not None:
                # Has it been delivered?
                if self.cargo_assignments[a] not in [c.id for c in state["active_cargo"]]:
                    # Unassign it
                    self.cargo_assignments[a] = None

            # If the airplane needs a new assignment...
            if pending_cargo and self.cargo_assignments[a] is None:
                # Check if there is any cargo needing to be delivered that is not assigned yet
                cargo_info = pending_cargo[self._np_random.choice(range(len(pending_cargo)))]

                if cargo_info.location != NOAIRPORT_ID:
                    full_delivery_path = self._full_delivery_paths[cargo_info.id]
                    try:
                        # Check if we should pick up this cargo...
                        # Can we make any progress after pickup? If not, move on to next cargo...
                        if not self.multidigraph.has_edge(cargo_info.location, full_delivery_path[0], obs[a]['plane_type']):
                            continue

                        # Generate a pickup path. If we can't reach the cargo, this will throw an exception
                        path = oh.get_lowest_cost_path(state, obs[a]["current_airport"],
                                                               cargo_info.location,
                                                               obs[a]["plane_type"])

                        # If we made it here, we should pick this cargo up. Complete the path to the dropoff location...
                        # Follow the full delivery path until we can't go further.
                        while full_delivery_path and self.multidigraph.has_edge(path[-1], full_delivery_path[0], obs[a]['plane_type']):
                            path.append(full_delivery_path.pop(0))

                        # Make the assignment
                        self.path[a] = path
                        self.cargo_assignments[a] = cargo_info.id
                        pending_cargo.remove(cargo_info)

                        # Once we have found a cargo and assigned it, we can break out of this loop.
                        break

                    except NetworkXNoPath as e:
                        # If there is no path, to pick up and/or deliver, don't complete the assignment
                        pass

            # If the plane is idle, assign a new action
            # We only assign an action while idle. If we try to assign a new action while the plane is not idle, we could hit on some glitches
            if oh.is_airplane_idle(obs[a]):
                actions[a] = {"process": 1,
                              "cargo_to_load": set(),
                              "cargo_to_unload": set(),
                              "destination": NOAIRPORT_ID}

                # If we have a path to follow, set next destination and takeoff when ready
                if self.path[a]:
                    next_destination = self.path[a][0]

                    # Have we arrived at the next destination? Pop that one off and set the next one in the list.
                    if obs[a]["current_airport"] == next_destination:
                        self.path[a].pop(0)
                        if self.path[a]:
                            next_destination = self.path[a][0]
                        else:
                            next_destination = NOAIRPORT_ID

                    actions[a]["destination"] = next_destination

                # Get info about the currently assigned cargo
                ca = oh.get_active_cargo_info(state, self.cargo_assignments[a])
                # If cargo is assigned

                if ca is not None:
                    if ca.id in obs[a]["cargo_onboard"]:
                        # If you're at final destination unload cargo or you have reached the end of your possible paths
                        if ca.destination == obs[a]['current_airport'] or not self.path[a]:
                            actions[a]["cargo_to_unload"].add(ca.id)
                            self.cargo_assignments[a] = None
                    elif ca.id in obs[a]['cargo_at_current_airport']:
                        actions[a]["cargo_to_load"].add(ca.id)

        return actions
