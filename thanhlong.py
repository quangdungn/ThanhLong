mport os
import SCL

model = SCL("scl.pt")

results = model.train(data="mydataset.yaml", epochs=100)
