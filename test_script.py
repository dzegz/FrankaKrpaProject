import unified_planning as up
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import *

import re

print(up.shortcuts.get_environment().factory.engines)
print("nesto2")
domain_file = "domain.pddl"
problem_file = "problem.pddl"

reader = PDDLReader()
print("nesto")
pddl_problem = reader.parse_problem(domain_file, problem_file)
print(pddl_problem.kind)

# disabling the credits
#up.shortcuts.get_environment().credits_stream = None

# solving the problem
#with OneshotPlanner(name='pyperplan') as planner:
with OneshotPlanner(problem_kind=pddl_problem.kind) as planner:
    result = planner.solve(pddl_problem)
    print("%s returned: %s" % (planner.name, result.plan))