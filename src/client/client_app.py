import torch
import torch.nn as nn
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from src.tasks import Interval, eICU, CPRD, Example

app = ClientApp()

@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the model on local data."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Client {msg.content['config']} starting train...")
    client_id = context.node_config["partition-id"]
    task_name = msg.content["config"]["task_name"]
    
    if task_name == "interval":
        task = Interval(config=msg.content["config"], device=device, cid=client_id)
    elif task_name == "cprd":
        task = CPRD(config=msg.content["config"], device=device, cid=client_id)
        print(f"Client {client_id} loading CPRD task...")
    elif task_name == "eicu":
        task = eICU(config=msg.content["config"], device=device, cid=client_id)
    else:
        task = Example(config=msg.content["config"], device=device, cid=client_id)        
                    

    task.set_models(msg.content["arrays"].to_torch_state_dict(), msg.content["icnn"].to_torch_state_dict())
   
    best_state_dict, omega = task.train()
    save_path = f"src/checkpoints/{task_name}_{client_id}.pt"
    
    torch.save(best_state_dict, save_path)
    print(f"Client {client_id} loading data...")
    
    model_record = ArrayRecord(best_state_dict)
    metrics = {
        "omega": omega,
    }
    
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """This method is required by the Flower framework but is not used for evaluation."""

    print(f"Client {context.node_id} starting evaluate...")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    client_id = context.node_config["partition-id"]

    task_name = msg.content["config"]["task_name"]

    if task_name == "interval":
        task = Interval(config=msg.content["config"], device=device, cid=client_id)
    elif task_name == "cprd":
        task = CPRD(config=msg.content["config"], device=device, cid=client_id)
    elif task_name == "eicu":
        task = eICU(config=msg.content["config"], device=device, cid=client_id)
    else:
        task = Example(config=msg.content["config"], device=device, cid=client_id)   

    task.set_models(msg.content["arrays"].to_torch_state_dict(), None)
    loss, metrics = task.validate()

    print(f"Client {context.node_id} evaluate: "
          f"Loss={loss:.4f}, "
          f"metrics={metrics}")
    

    metrics["omega"] = 0.1 # required field for Flower, but this is not used for evaluation
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    return Message(content=content, reply_to=msg)