from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random
import logging

# Execute the agent to play against medium protoss, with 16 step gap with 32 concurrent simulations, 256 total:
# python3 -m pysc2.bin.agent --map Simple64 --agent Kane-AI.KaneAI --use_feature_units --agent_race terran --difficulty medium --agent2_race protoss --step_mul 16 --max_episodes 8 --norender --parallel 32
#
# Right now, difficulty logging is updated manually within the environment config within sc2_env.py from psyc2's library

# Our debug logging config
logging.basicConfig(level=logging.DEBUG)

# To execute this agent in debug mode, use the string:
# python3 my_test_agents2.py --logtostderr true --stderrthreshold debug
# logtostderr/debug info optional

# This agent builds only marines and is optimized to produce as many as possible


class KaneAI(base_agent.BaseAgent):
    # Basic class template reference is from PySC2 itself - additional self.* are customized for Kane AI
    def __init__(self):
        super(KaneAI, self).__init__()
        self.enemy_spawn_location = None
        self.idle_scv_selected = False
        self.initial_exploration_complete = False
        self.attack_expansion_location = False
        self.attack_agent_expansion_location = False
        self.attack_counter = 0
        self.expansion_counter = 0
        # These last_command variables check to make sure we're not repetitively spamming the same command over and over again
        self.last_command_idle_scv = False
        self.last_command_marine_attack = False
    @property
    def name(self):
        return "KaneAI"

    # The following 3 functions are boiler plate / common across all PySC2 Agents
    # This function checks to see if the currently selected unit matches the one we're looking for
    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False
    
    # This function identifies all units of a specific desired type
    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    # This function returns a list of actions for a given unit
    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    # This is our function to identify mineral locations so we can reference them later
    def get_mineral_locations(self, obs):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == units.Neutral.MineralField]

    # This checks our current population/supply
    def get_supply_info(self, obs):
        """Returns current supply usage and limit."""
        return obs.observation.player.food_used, obs.observation.player.food_cap

    # This is where all of the agent's logic belongs
    # This code is executed for every step
    def step(self, obs):
        super(KaneAI, self).step(obs)

        # Checks to see if this is the first step of the game
        if obs.first():
            # Figure out where our home base is
            player_y, player_x = (
                obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
            # Modified Steven Brown's logic from the following repository to identify attack coordinates
            # https://raw.githubusercontent.com/skjb/pysc2-tutorial/master/Build%20a%20Zerg%20Bot/zerg_agent_step7.py
            # Essentially, on the Simple64 map that we're using, the opponent will always be in the opposite quadrant of the map AFAIK
            if obs.first():
                player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                      features.PlayerRelative.SELF).nonzero()
                xmean = player_x.mean()
                ymean = player_y.mean()

                if xmean <= 31 and ymean <= 31:
                    self.enemy_spawn_location = (42, 46)
                    self.enemy_expansion_location = (18, 46)
                    self.agent_expansion_location = (42, 22)
                else:
                    self.enemy_spawn_location = (18, 22)
                    self.enemy_expansion_location = (42, 22)
                    self.agent_expansion_location = (18, 46)

        # This is our idle SCV work assignment queue
        # Needs to be at the top as other actions tend to take precedence and as such, no work is every assigned to idle SCVs
        if self.idle_scv_selected and not self.last_command_idle_scv:
            self.idle_scv_selected = False
            self.last_command_idle_scv = True
            self.last_command_marine_attack = False
            # print("IDLE SCV CODE")
            # check to see if we have valid selections (IE, not null)
            selected = obs.observation.single_select if len(
                obs.observation.single_select) > 0 else obs.observation.multi_select
            if selected.size > 0:
                # Check if Harvest_Gather_SCV_screen action is available
                minerals = self.get_mineral_locations(obs)
                if minerals:
                    # Select a random mineral patch
                    mineral_patch = random.choice(minerals)
                    if actions.FUNCTIONS.Harvest_Gather_screen.id in obs.observation.available_actions:
                        # print("Sending SCV to mine minerals")
                        return actions.FUNCTIONS.Harvest_Gather_screen("now", (mineral_patch.x, mineral_patch.y))
                    # else:
                    #     print("Harvest action is not available")

        # Identify how many minerals we currently have
        # This is referenced pretty extensively throughout (IE, when building units/buildings)
        mineralCount = obs.observation.player.minerals

        # This identifies all our workers (SCVs in our case but could be drone or probe)
        workers = self.get_units_by_type(obs, units.Terran.SCV)
        # This identifies our idle SCVs (order length = 0)
        idle_workers = [scv for scv in workers if scv.order_length == 0]

        # Identifies any marines we have built currently
        marines = self.get_units_by_type(obs, units.Terran.Marine)

        # Attacking Logic

        # If we have more than 12 marines, send them to the enemy spawn location
        # print("Length of marine list is:", len(marines))
        if len(marines) >= 18 and not self.last_command_marine_attack:
            # If we have marines already selected...
            if self.unit_type_is_selected(obs, units.Terran.Marine):
                # print("We're in marine attack code")
                # print("attack_counter: ", self.attack_counter, "expansion_counter:", self.expansion_counter)
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id) and self.attack_counter < 3:
                    # print("Issuing attack minimap command")
                    self.last_command_marine_attack = True
                    self.last_command_idle_scv = False
                    self.attack_expansion_location = True
                    # Increment our counter so that on the third iteration through we'll go after the expansion
                    self.attack_counter += 1
                    return actions.FUNCTIONS.Attack_minimap("now", self.enemy_spawn_location)
                # We're now attacking the expansion location
                elif self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id) and self.attack_expansion_location and self.expansion_counter < 1:
                    self.last_command_marine_attack = True
                    self.last_command_idle_scv = False
                    self.attack_agent_expansion_location = True
                    self.expansion_counter += 1
                    return actions.FUNCTIONS.Attack_minimap("now", self.enemy_expansion_location)
                # If the game is still going, that probably means the enemy has also created a third expansion in our location...
                if self.attack_agent_expansion_location and self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id) and self.attack_counter >= 2 and self.expansion_counter >= 1:
                    self.last_command_marine_attack = True
                    self.last_command_idle_scv = False
                    self.attack_counter = 0
                    self.expansion_counter = 0
                    return actions.FUNCTIONS.Attack_minimap("now", self.agent_expansion_location)

            # If we don't have any selected, do a group select so we can issue an attack order next step
            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")
            # else:
            #     print("Cannot issue select_army command")

        # Select any idle SCV and in the next step tell them to mine minerals.
        # Since "selecting" a unit consumes a step, we have to do this in two pieces of code
        # Step one (this one), select a unit
        # Step two, (`if self.idle_scv_selected`), do something with it
        if idle_workers and not self.last_command_idle_scv:
            scv = random.choice(idle_workers)
            if not self.idle_scv_selected:
                # Sometimes the coordinates this returns are out of bounds and causes a crash, hence the check >=0
                if actions.FUNCTIONS.select_point.id in obs.observation.available_actions and scv.x >= 0 and scv.y >= 0:
                    self.idle_scv_selected = True
                    # print(f"Selecting SCV at ({scv.x}, {scv.y})")
                    return actions.FUNCTIONS.select_point("select", (scv.x, scv.y))
                # else:
                #     print("select_point action is not available.")

        # Checks our supply and builds supply depots when necessary
        supply_used, supply_cap = self.get_supply_info(obs)
        # Builds supply when we have less than 3 available and when we have more than 100 minerals
        if (supply_cap - supply_used) <= 3:
            # doing nested mineral check for performance reasons
            if mineralCount >= 100:
                # Check if a Supply Depot is under construction
                supply_depots = self.get_units_by_type(
                    obs, units.Terran.SupplyDepot)
                supply_depot_under_construction = any(
                    [depot.build_progress < 100 for depot in supply_depots])
                # If a Supply Depot is under construction, skip the building part
                if supply_depot_under_construction:
                    return actions.FUNCTIONS.no_op()
                # select a random SCV
                if len(workers) > 0:
                    scv = random.choice(workers)
                    # print("chose SCV: ", scv)
                    if not self.unit_type_is_selected(obs, units.Terran.SCV):
                        # If an SCV is not selected, select one
                        if self.can_do(obs, actions.FUNCTIONS.select_point.id):
                            if scv.x >= 0 and scv.y >= 0:
                                return actions.FUNCTIONS.select_point("select_all_type", (scv.x, scv.y))
                            else:
                                print("Invalid SCV coordinates, try again")
                    else:
                        # We already have one, let's build a supply depot
                        if self.can_do(obs, actions.FUNCTIONS.Build_SupplyDepot_screen.id):
                            x = random.randint(6, 76)
                            y = random.randint(6, 76)
                            return actions.FUNCTIONS.Build_SupplyDepot_screen("now", (x, y))
        # Early Exploration
        # Check to see if we're executed the initial exploration or not (global variable)
        # A lot of the SCV code is boilerplate - also present in Steven Brown's repository referenced above
        if not self.initial_exploration_complete:
            if len(obs.observation.single_select) == 0 or obs.observation.single_select[0].unit_type != 45:
                scvs = self.get_units_by_type(obs, units.Terran.SCV)
                if scvs:
                    scv = random.choice(scvs)
                    if scv.x >= 0 and scv.y >= 0:
                        return actions.FUNCTIONS.select_point("select", (scv.x, scv.y))
                    else:
                        print("Invalid SCV coordinates, try again")
            if actions.FUNCTIONS.Move_minimap.id in obs.observation.available_actions:
                # We can now set the global variable to false so we don't run this code again
                target = self.enemy_spawn_location
                self.initial_exploration_complete = True
                return actions.FUNCTIONS.Move_minimap("now", target)

        # Military Units
        # Barracks Generator
        # More or less boiler plate code for building these, same Steven Brown repository as elsewhere
        barracks = self.get_units_by_type(obs, units.Terran.Barracks)
        # if we have less than 2 barracks, enough minerals to build one, and we've got enough baseline of workers
        # Or if we've progressed to > 12 marines we need more barracks
        # Build a barracks
        if len(barracks) < 4:
            # Conditional checks are nested to make lookups less expensive
            if mineralCount >= 150 and len(workers) > 10:
                if self.unit_type_is_selected(obs, units.Terran.SCV):
                    if self.can_do(obs, actions.FUNCTIONS.Build_Barracks_screen.id):
                        x = random.randint(6, 76)
                        y = random.randint(6, 76)
                        return actions.FUNCTIONS.Build_Barracks_screen("now", (x, y))

        # Marine Creation
        # If we have enough barracks to begin building marines at a reasonable rate
        if len(barracks) >= 2:
            # If we have a barracks selected
            if self.unit_type_is_selected(obs, units.Terran.Barracks):
                if len(marines) <= 80:
                    if self.can_do(obs, actions.FUNCTIONS.Train_Marine_quick.id):
                        return actions.FUNCTIONS.Train_Marine_quick("now")

            # We hit this code if a barracks wasn't selected last step
            # In case it wasn't...we select it so we can build marines next step
            selected_barracks = random.choice(barracks)
            return actions.FUNCTIONS.select_point("select_all_type", (selected_barracks.x, selected_barracks.y))

        # If there are fewer than 16 SCVs, train an SCV at the Command Center
        # As expected, a lot of this is also boilerplate for SCVs and can be found in Steven Browns repository referenced above
        command_centers = self.get_units_by_type(
            obs, units.Terran.CommandCenter)
        if len(workers) < 16 and command_centers:
            command_center = random.choice(command_centers)
            if not self.unit_type_is_selected(obs, units.Terran.CommandCenter):
                # If a CommandCenter is not selected, select one
                if self.can_do(obs, actions.FUNCTIONS.select_point.id):
                    return actions.FUNCTIONS.select_point("select_all_type", (command_center.x, command_center.y))
            else:
                if self.can_do(obs, actions.FUNCTIONS.Train_SCV_quick.id):
                    return actions.FUNCTIONS.Train_SCV_quick("now", command_center.tag)

        # If we reach here, we choose to do a NOP for this step
        return actions.FUNCTIONS.no_op()
