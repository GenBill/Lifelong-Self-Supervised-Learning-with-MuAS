from agent.train_rota import rotatrain
from agent.train_patch import patchtrain
from agent.train_jigpa import jigpatrain
from agent.train_jigro import jigrotrain

def Step(
    model_ft, fc_layer, powerword,
    myloader, criterion, optimizer, scheduler, 
    device, out_dir, file, saveinterval, num_epochs
):
    # default num_step == 0,1,2,3
    if powerword=='rota':
        model_ft, fc_layer = rotatrain(
            model_ft, fc_layer, 
            myloader, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, num_epochs
        )
        return model_ft, fc_layer

    elif powerword=='patch':
        model_ft, fc_layer = patchtrain(
            model_ft, fc_layer, 
            myloader, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, num_epochs
        )
        return model_ft, fc_layer

    elif powerword=='jigpa':
        model_ft, fc_layer = jigpatrain(
            model_ft, fc_layer, 
            myloader, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, num_epochs
        )
        return model_ft, fc_layer

    elif powerword=='jigro':
        model_ft, fc_layer = jigrotrain(
            model_ft, fc_layer, 
            myloader, criterion, optimizer, scheduler, 
            device, out_dir, file, saveinterval, num_epochs
        )
        return model_ft, fc_layer

    # default num_step == 0,1,2,3
    # else return 0