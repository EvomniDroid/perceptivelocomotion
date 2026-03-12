with open('/home/zh/isaac/instinctlab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_shadowing_cfg.py', 'r') as f:
    content = f.read()

content = content.replace("self.scene.motion_reference.motion_buffers[\"AMASSMotion\"] = AMASSMotionCfg()", "self.scene.motion_reference.motion_buffers[\"AMASSMotion\"] = AMASSMotionCfg()\n            MOTION_NAME = \"AMASSMotion\"")
content = content.replace("PLANE_TERRAIN = False", "PLANE_TERRAIN = True")

with open('/home/zh/isaac/instinctlab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_shadowing_cfg.py', 'w') as f:
    f.write(content)
