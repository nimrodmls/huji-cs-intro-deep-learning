import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld
import matplotlib.pyplot as plt

batch_size = 32
output_size = 2
hidden_size = 64        # to experiment with

run_recurrent = False    # else run Token-wise MLP
use_RNN = False          # otherwise GRU
atten_size = 2 # Restricted self attention configured to context of 2 words to the left & right each

reload_model = False
num_epochs = 10
learning_rate = 0.001
test_interval = 50

# Loading sataset, use toy = True for obtaining a smaller dataset
train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size)

# Setting device to be used across the board
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def simplify_labels(labels):
    """
    Producing the vector of positives-negatives, where 1 is positive and 0 is negative
    from the original labels of [1,0] (Positive) & [0,1] (Negative)
    """
    return (labels == torch.tensor([1.0, 0.0], device=device)).all(axis=1).type(torch.int)

def get_prediction_vector(output_vec):
    """
    Producing the vector of positives-negatives, where 1 is positive and 0 is negative
    from the output vector of a model
    """
    # E.g. where the first element is the highest, it is a positive review
    return torch.where(torch.argmax(output_vec, axis=1) == 0, 1.0, 0.0)

# Special matrix multipication layer (like torch.Linear but can operate on arbitrary sized
# tensors and considers its last two indices as the matrix.)
class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias = True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels,out_channels)), requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1,1,out_channels), requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):        
        x = torch.matmul(x,self.matrix) 
        if self.use_bias:
            x = x + self.bias 
        return x
        
class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()

        self.hidden_size = hidden_size
        # TODO: Remove
        #self.sigmoid = torch.sigmoid

        # TODO: Experiment with in2out
        # Cell FC, generating hidden state for the next step
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        # nn.Sequential(
        #     nn.Linear(input_size + hidden_size, hidden_size), # hidden state for next step
        #     nn.Tanh()
        # )

        # Output layer for the current step
        self.hidden2out = nn.Linear(input_size + hidden_size, output_size)

    def name(self):
        return "RNN"
    
    def uid(self):
        return f'{self.name()}_{self.hidden_size}'

    def forward(self, x, hidden_state):
        # Concatenating input and hidden state for input to the RNN cell
        layer_in = torch.cat((x, hidden_state), 1)
        
        # Feeding
        hidden = self.in2hidden(layer_in)
        output = self.hidden2out(layer_in)

        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size).to(device)

class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size

        # GRU Cell weights
        # Reset gate (r)
        self.reset = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # Update gate (z)
        self.update = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # Hidden Candidate (h)
        self.hidden_cand = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )

        self.output_layer = nn.Linear(hidden_size, output_size)

    def name(self):
        return "GRU"
    
    def uid(self):
        return f'{self.name()}_{self.hidden_size}'

    def forward(self, x, hidden_state):
        layer_in = torch.cat((x, hidden_state), 1)

        # Intermediary calculations of the GRU cell
        update_out = self.update(layer_in)
        hidden_candidate = self.hidden_cand(torch.cat((x, self.reset(layer_in) * hidden_state), 1))

        # Determining the new hidden state & the output accordingly
        new_hidden_state = (hidden_state * update_out) + ((1 - update_out) * hidden_candidate)
        output = self.output_layer(new_hidden_state)

        return output, new_hidden_state

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size).to(device)

class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()

        # Token-wise MLP network weights
        self.fc = nn.Sequential(
            MatMul(input_size, hidden_size),
            nn.ReLU(),
            MatMul(hidden_size, output_size)
        )

    def name(self):
        return "MLP"
    
    def uid(self):
        return self.name()

    def forward(self, x):
        return self.fc(x)

class ExLRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExLRestSelfAtten, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.softmax = torch.nn.Softmax(2)
        
        # Token-wise MLP + Restricted Attention network implementation

        self.input_layer = nn.Sequential(
            MatMul(input_size, hidden_size),
            nn.ReLU(),
        )

        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False)

        # The ouput layer is computing the sub-prediction scores, as in the original MLP
        self.output_layer = nn.Sequential(
            MatMul(hidden_size, output_size)
        )

    def name(self):
        return "MLP_atten"
    
    def uid(self):
        return self.name()

    def forward(self, x):
        x = self.input_layer(x)

        # generating x in offsets between -atten_size and atten_size 
        # with zero padding at the ends
        padded = pad(x,(0,0,atten_size,atten_size,0,0))

        x_nei = []
        for k in range(-atten_size,atten_size+1):
            x_nei.append(torch.roll(padded, k, 1))

        x_nei = torch.stack(x_nei,2)
        x_nei = x_nei[:,atten_size:-atten_size,:]
        
        # x_nei has an additional axis that corresponds to the offset

        ## Applying attention layer

        query = self.W_q(x)
        keys = self.W_k(x_nei)
        vals = self.W_v(x_nei)

        keys_T = keys.transpose(2, 3)
        # Calculating the D parameter, note the addition of extra axis for the offset
        d = torch.matmul(query[:, :, None, :], keys_T) / self.sqrt_hidden_size
        atten_weights = self.softmax(d)

        return self.output_layer(torch.matmul(atten_weights, vals)).squeeze(), atten_weights

def print_review(rev_text, sbs1, sbs2, lbl1, lbl2):
    """
    Prints portion of the review (20-30 first words), with the sub-scores each word obtained
    prints also the final scores, the softmaxed prediction values and the true label values
    """
    word_cnt = 30
    print(f'Review: {" ".join(rev_text[:word_cnt])}')
    sbs1 = nn.functional.sigmoid(torch.tensor(sbs1))
    sbs2 = nn.functional.sigmoid(torch.tensor(sbs2))
    print('Sub-scores:')
    for word, s1, s2 in zip(rev_text[:word_cnt], sbs1[:word_cnt], sbs2[:word_cnt]):
        print(f'\t{word}: Positive: {s1:.4f}, Negative: {s2:.4f}')
    final_scores = torch.tensor([torch.mean(sbs1), torch.mean(sbs2)])
    print(f'Final Scores: Positive: {final_scores[0]:.4f}, Negative: {final_scores[1]:.4f}')
    get_label = lambda pred: 'Positive' if pred[0] == 1.0 else 'Negative'
    print(f'Prediction: {get_label(get_prediction_vector(final_scores[None, :]))}, '
          f'True: {get_label(simplify_labels(torch.tensor([[lbl1, lbl2]], device=device)))}')

def model_experiment(model):
    model.to(device)
    print("Using model: " + model.name())

    if reload_model:
        print("Reloading model")
        model.load_state_dict(torch.load(model.name() + ".pth"))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []
    true_positives = 0
    false_negatives = 0

    # training steps in which a test step is executed every test_interval
    for epoch in range(num_epochs):

        itr = 0 # iteration counter within each epoch

        # Accumalating correct predictions for accuracy calculation
        correct_train = 0
        total_train = 0
        correct_test = 0
        total_test = 0
        total_test_batches = 0
        train_ep_loss = 0
        test_ep_loss = 0

        for labels, reviews, reviews_text in train_dataset:   # getting training batches
            labels.to(device)
            reviews.to(device)

            itr = itr + 1

            if (itr + 1) % test_interval == 0:
                test_iter = True
                labels, reviews, reviews_text = next(iter(test_dataset)) # get a test batch 
                labels.to(device)
                reviews.to(device)
            else:
                test_iter = False

            # Recurrent nets (RNN/GRU)
            if model.name() in ["RNN", "GRU"]:
                hidden_state = model.init_hidden(int(labels.shape[0]))

                for i in range(num_words):
                    output, hidden_state = model(reviews[:,i,:], hidden_state)  # HIDE
            else:  # Token-wise networks (MLP / MLP + Atten.)
                sub_score = []
                if model.name() == 'MLP_atten': # MLP + atten
                    sub_score, atten_weights = model(reviews)
                else: # MLP
                    sub_score = model(reviews)

                output = torch.mean(sub_score, 1)
                
            # cross-entropy loss
            loss = criterion(output, labels)

            # optimize in training iterations
            if not test_iter:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Simplified, binary labels & predictions representation
            simple_labels = simplify_labels(labels)
            pred_vec = get_prediction_vector(output)
            correct_preds = (simple_labels == pred_vec).sum().item()
            
            if test_iter:
                test_ep_loss += loss.item()
                correct_test += correct_preds
                total_test += labels.shape[0]
                total_test_batches += 1
                true_positives += pred_vec.sum().item()
                false_negatives += simple_labels[pred_vec == 0.0].sum().item() # Counting the amount of misclassifications
            else:
                train_ep_loss += loss.item()
                correct_train += correct_preds
                total_train += labels.shape[0]

            if test_iter:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{itr + 1}/{len(train_dataset)}], "
                    f"Train Loss: {loss.item():.4f}, "
                    f"Test Loss: {loss.item():.4f}, "
                    f"Train Accuracy: {correct_train / total_train:.4f}, "
                    f"Test Accuracy: {correct_test / total_test:.4f}"
                )

                if model.name() not in ['RNN', 'GRU']:
                    nump_subs = sub_score.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()
                    #print_review(reviews_text[0], nump_subs[0,:,0], nump_subs[0,:,1], labels[0,0], labels[0,1])

                # saving the model
                torch.save(model, f'model_{model.uid()}.pth')

        train_losses.append(train_ep_loss / len(train_dataset))
        test_losses.append(test_ep_loss / total_test_batches) # For now using only one batch at a time
        train_accuracy.append(correct_train / total_train)
        test_accuracy.append(correct_test / total_test)

    print(f'{model.name()} - Recall: {true_positives / (true_positives + false_negatives):.4f}, '
          f'Test Accuracy: {test_accuracy[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')
    # Saving the final results
    torch.save(train_losses, f"train_losses_{model.uid()}.pth")
    torch.save(test_losses, f"test_losses_{model.uid()}.pth")
    torch.save(train_accuracy, f"train_accuracy_{model.uid()}.pth")
    torch.save(test_accuracy, f"test_accuracy_{model.uid()}.pth")

def evaluate_model(model):
    """
    Evaluating the model with test samples and finding samples which are 
    True-Positive, True-Negative, False-Positive & False-Negative
    """
    model.to(device)
    model.eval()

    true_pos = None
    false_pos = None
    true_neg = None
    false_neg = None
    
    with torch.no_grad():
        for labels, reviews, reviews_text in test_dataset:
            labels.to(device)
            reviews.to(device)

            if model.name() in ["RNN", "GRU"]:
                hidden_state = model.init_hidden(int(labels.shape[0]))

                for i in range(num_words):
                    output, hidden_state = model(reviews[:,i,:], hidden_state)
            else:
                sub_score = []
                if model.name() == 'MLP_atten':
                    sub_score, atten_weights = model(reviews)
                else:
                    sub_score = model(reviews)

                output = torch.mean(sub_score, 1)

            simple_labels = simplify_labels(labels)
            pred_vec = get_prediction_vector(output)

            for i in range(labels.shape[0]):
                if simple_labels[i] == 1.0 and pred_vec[i] == 1.0:
                    true_pos = (reviews_text[i], sub_score[i,:,0], sub_score[i,:,1], 1, 0)
                elif simple_labels[i] == 0.0 and pred_vec[i] == 0.0:
                    true_neg = (reviews_text[i], sub_score[i,:,0], sub_score[i,:,1], 0, 1)
                elif simple_labels[i] == 0.0 and pred_vec[i] == 1.0:
                    false_pos = (reviews_text[i], sub_score[i,:,0], sub_score[i,:,1], 0, 1)
                elif simple_labels[i] == 1.0 and pred_vec[i] == 0.0:
                    false_neg = (reviews_text[i], sub_score[i,:,0], sub_score[i,:,1], 1, 0)

    print("## True Positive")
    print_review(*true_pos)
    print("## False Positive")
    print_review(*false_pos)
    print("## True Negative")
    print_review(*true_neg)
    print("## False Negative")
    print_review(*false_neg)

def plot_results(data, title, xlabel, ylabel, filename):
    """
    """
    plt.figure()
    for x, description in data:
        x = torch.load(x)
        plt.plot(x, label=description)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename)

def main():
    # Running experimentation for each model superclass
    for model in [
                  #ExRNN(input_size, output_size, hidden_size=64), 
                  #ExRNN(input_size, output_size, hidden_size=128), 
                  #ExGRU(input_size, output_size, hidden_size=64),
                  #ExGRU(input_size, output_size, hidden_size=128),
                  #ExMLP(input_size, output_size, hidden_size), 
                  #ExLRestSelfAtten(input_size, output_size, hidden_size)
                ]:
        model_experiment(model)

    # Evaluating models on specific terms
    print("Evaluating MLP model...")
    evaluate_model(torch.load('model_MLP.pth'))

    print("Evaluating MLP + Attention model...")
    evaluate_model(torch.load('model_MLP_atten.pth'))

    # Plotting the results
    plot_results([('train_losses_RNN_64.pth', "RNN (64) - Train"),
                  ('test_losses_RNN_64.pth', "RNN (64) - Test"),
                  ("train_losses_GRU_64.pth", "GRU (64) - Train"),
                  ('test_losses_GRU_64.pth', "GRU (64) - Test"),
                  ("train_losses_RNN_128.pth", "RNN (128) - Train"), 
                  ('test_losses_RNN_128.pth', "RNN (128) - Test"),
                  ("train_losses_GRU_128.pth", "GRU (128) - Train"),
                  ('test_losses_GRU_128.pth', "GRU (128) - Test")], 
                  "Losses: GRU vs. RNN", "Epochs", "Loss", "losses_RNN_GRU.pdf")
    plot_results([('train_accuracy_RNN_64.pth', "RNN (64) - Train"),
                  ('test_accuracy_RNN_64.pth', "RNN (64) - Test"),
                  ("train_accuracy_GRU_64.pth", "GRU (64) - Train"),
                  ('test_accuracy_GRU_64.pth', "GRU (64) - Test"),
                  ("train_accuracy_RNN_128.pth", "RNN (128) - Train"), 
                  ("test_accuracy_RNN_128.pth", "RNN (128) - Test"),
                  ("train_accuracy_GRU_128.pth", "GRU (128) - Train"),
                  ("test_accuracy_GRU_128.pth", "GRU (128) - Test")],
                  "Accuracy: GRU vs. RNN", "Epochs", "Accuracy", "accuracy_RNN_GRU.pdf")
    plot_results([("train_losses_MLP.pth", "MLP - Train"), 
                  ('test_losses_MLP.pth', "MLP - Test")], 
                  "Losses: MLP", "Epochs", "Loss", "losses_MLP.pdf")
    plot_results([("train_accuracy_MLP.pth", "MLP - Train"), 
                  ("test_accuracy_MLP.pth", "MLP - Test")],
                   "Accuracy: MLP", "Epochs", "Accuracy", "accuracy_MLP.pdf")
    plot_results([("train_losses_MLP_atten.pth", "MLP + Atten - Train"),
                  ('test_losses_MLP_atten.pth', "MLP + Atten - Test")], 
                  "Losses: MLP + Atten", "Epochs", "Loss", "losses_MLP_atten.pdf")
    plot_results([("train_accuracy_MLP_atten.pth", "MLP + Atten - Train"), 
                  ("test_accuracy_MLP_atten.pth", "MLP + Atten - Test")],
                  "Accuracy: MLP + Atten", "Epochs", "Accuracy", "accuracy_MLP_atten.pdf")

if __name__ == '__main__':
    main()