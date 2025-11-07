
from prettytable import PrettyTable

def count_parameters(model, verbose=0):

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    if verbose == 1:
        print(table)
        print(f"Total Trainable Params: {total_params}")
    return total_params