import argparse
import sys
import matplotlib
# This is needed to save images 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_train_val_losses(filename):
    # Parse the train and val losses one line at a time.
    import re
    # regexes to find train and val losses on a line
    float_regex = r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'
    train_loss_re = re.compile('.*Train Loss: ({})'.format(float_regex))
    val_loss_re = re.compile('.*Val Loss: ({})'.format(float_regex))
    val_acc_re = re.compile('.*Val Acc: ({})'.format(float_regex))
    # extract one loss for each logged iteration
    train_losses = []
    val_losses = []
    val_accs = []
    # NOTE: You may need to change this file name.
    with open(filename, 'r') as f:
        for line in f:
            train_match = train_loss_re.match(line)
            val_match = val_loss_re.match(line)
            val_acc_match = val_acc_re.match(line)
            if train_match:
                train_losses.append(float(train_match.group(1)))
            if val_match:
                val_losses.append(float(val_match.group(1)))
            if val_acc_match:
                val_accs.append(float(val_acc_match.group(1)))
    return train_losses, val_losses, val_accs

def save_figs(model_name, filename, train_losses, val_losses, val_accs):
    fig = plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.title('{0} Learning Curve'.format(model_name))
    plt.ylabel('loss')
    plt.legend()
    fig.savefig('{0}_lossvstrain.png'.format(filename))

    fig = plt.figure()
    plt.plot(val_accs, label='val')
    plt.title('{0} Validation Accuracy During Training'.format(model_name))
    plt.ylabel('accuracy')
    plt.legend()
    fig.savefig('{0}_valaccuracy.png'.format(filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Generation Tool")

    parser.add_argument('--log-file', type=str, help="path to log file")
    parser.add_argument('--file-name', type=str, help="output file name")
    parser.add_argument('--model-name', type=str, help="model name")

    if not args.log_file:
        print("Log file must be specified")
        sys.exit(1)

    if not args.file_name and not args.model_name:
        print("Either the file name or the model name must be specified.")
        sys.exit(1)

    args = parser.parse_args()
    train_losses, val_losses, val_accs = parse_train_val_losses(args.log_file)
    
    filename = args.file_name
    model_name = args.model_name
    if not filename:
        filename = model_name
    elif not model_name:
        model_name = filename

    save_figs(model_name, filename, train_losses, val_losses, val_accs)