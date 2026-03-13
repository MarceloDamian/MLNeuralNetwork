import csv, numpy as np, pandas as pd
from sklearn.utils import shuffle
from scipy.special import log_softmax

# ─────────────────────────────────────────────
#  Global state (set by train() before use)
# ─────────────────────────────────────────────
Nodes = None
Iter_Index = None
Hiddenlayers = None
WandB = None
Prevvalues = None
Altvalues = None
NWandB = None
OptWandB = None
LayerWandB = None
ACTIV = None
DERIV = None


class Sequential():

    # ── Initialization ──────────────────────────────────────────────────────

    def init_var(self):
        return [0] * (Iter_Index * 2)

    def init_hiddenlayers(self):
        return np.empty((len(Nodes) - 2), dtype=object)

    def weightsandbiases(self):
        """He initialization for weights; zero vectors for biases."""
        dynamicwandb = []
        for i in range(Iter_Index):
            np.random.seed(i)
            w = np.random.uniform(size=(Nodes[i+1], Nodes[i]), low=-1, high=1) * np.sqrt(2 / Nodes[i])
            b = np.zeros((Nodes[i+1], 1))
            dynamicwandb += [w, b]
        return dynamicwandb

    # ── Data loading ─────────────────────────────────────────────────────────

    def load_data(self, filepath):
        """Loads and shuffles CSV. Returns (X, y) as numpy arrays."""
        data = pd.read_csv(filepath)
        data = shuffle(data, random_state=1).reset_index(drop=True)
        arr = np.array(data)
        X = arr[:, 1:] / 255.0        # normalize pixels to [0, 1]
        y = arr[:, 0].astype(int)     # labels
        return X, y

    # ── Forward pass ─────────────────────────────────────────────────────────

    def Linear(self, layer2nodes, layer3nodes, arraymapped, w_, b_):
        npinsertarr = np.array(arraymapped).reshape(layer2nodes, 1)
        return (np.dot(w_, npinsertarr) + b_).reshape(layer3nodes)

    def LeakyReLU(self, z, alpha=0.01):
        return np.maximum(alpha, z)

    def D_LeakyReLU(self, z, alpha=0.01):
        return np.where(z > 0, 1.0, alpha)

    def Softmax(self, z):
        exp = np.exp(z - np.max(z))
        self.softmaxlist = exp / np.sum(exp)
        return self.softmaxlist

    def forward(self, x):
        """Full forward pass through all hidden layers then output softmax."""
        current = x
        for i in range(Hiddenlayers):
            z = self.Linear(Nodes[i], Nodes[i+1], current, WandB[i*2], WandB[i*2+1])
            ACTIV[i] = self.LeakyReLU(z)
            DERIV[i] = self.D_LeakyReLU(z)  # pass pre-activation z, not post-activation output
            current = ACTIV[i]
        z_out = self.Linear(Nodes[-2], Nodes[-1], current, WandB[-2], WandB[-1])
        self.scaledarray = x  # store input for backprop
        return self.Softmax(z_out)

    # ── Loss & scoring ────────────────────────────────────────────────────────

    def Hotencode(self, desired):
        return np.eye(Nodes[-1], dtype="float")[desired]

    def C_CrossEntropyLoss(self, label):
        self.label = label
        targethotencode = self.Hotencode(label)
        return np.sum(-targethotencode * log_softmax(self.softmaxlist))

    def Score(self, label):
        """Returns 1 if prediction is correct, else 0."""
        return int(np.argmax(self.softmaxlist) == label)

    # ── Backward pass ────────────────────────────────────────────────────────

    def D_Softmax(self):
        soft = self.softmaxlist.reshape(Nodes[-1], 1)
        self.partials = np.diagflat(soft) - np.dot(soft, soft.T)
        return self.partials

    def D_CCELoss(self, label):
        targethotencode = self.Hotencode(label)
        safe_softmax = np.where(self.softmaxlist == 0.0, 0.01, self.softmaxlist)
        self.crossder = -targethotencode / safe_softmax
        return self.crossder

    def D_CCE_and_Softmax(self, label):
        self.loss = self.D_Softmax() @ self.D_CCELoss(label)
        return self.loss

    def helper_loss_prop(self, floor):
        ArrayIndex = Iter_Index * 2 - 2
        wloss = np.dot(
            self.D_CCE_and_Softmax(self.label).reshape(1, Nodes[Iter_Index]),
            WandB[ArrayIndex].reshape(Nodes[Iter_Index], Nodes[Iter_Index-1])
        ) * DERIV[len(Nodes)-3]
        for decrement in range(Iter_Index-1, floor, -1):
            wloss = np.dot(
                wloss.reshape(1, Nodes[decrement]),
                WandB[ArrayIndex-2].reshape(Nodes[decrement], Nodes[decrement-1])
            ) * DERIV[decrement - 2]
            ArrayIndex -= 2
        return wloss

    def helper_wandb_prop(self, start, clock, startminusclock, array, wloss):
        LayerWandB[start]   = wloss.reshape(Nodes[startminusclock], 1) * array.reshape(1, Nodes[clock+1])
        LayerWandB[start+1] = np.sum(LayerWandB[start], 1).reshape(Nodes[startminusclock], 1)
        return LayerWandB[start], LayerWandB[start+1]

    def Backward_Prop(self):
        clock, floor = 0, 2
        LayerWandB[0], LayerWandB[1] = self.helper_wandb_prop(
            0, -1, 1, self.scaledarray, self.helper_loss_prop(1)
        )
        for next_ in range(2, Iter_Index * 2, 2):
            if len(Nodes) != 3:
                wloss = self.helper_loss_prop(floor)
            if next_ == Iter_Index * 2 - 2:
                wloss = self.loss.reshape(Nodes[-1], 1)
            LayerWandB[next_], LayerWandB[next_+1] = self.helper_wandb_prop(
                next_, clock, int(next_ - clock), ACTIV[clock], wloss
            )
            floor += 1
            clock += 1
        return LayerWandB[0: Iter_Index * 2]

    # ── Optimization ─────────────────────────────────────────────────────────

    def GradientDescentWithMomentum(self, mu, lr, is_first):
        """SGD + Momentum: velocity = mu * prev_velocity + lr * gradient"""
        for i in range(len(Nodes) + Hiddenlayers):
            if is_first:
                Altvalues[i] = lr * NWandB[i]
            else:
                Altvalues[i] = mu * Prevvalues[i] + lr * NWandB[i]
            OptWandB[i] = WandB[i] - Altvalues[i]
        return OptWandB, Altvalues

    def get_lr(self, base_lr, epoch, decay_rate=0.95):
        """Exponential learning rate decay — applied once per epoch."""
        return base_lr * (decay_rate ** epoch)

    # ── Batch gradient helpers ────────────────────────────────────────────────

    def accumulate_batch_gradients(self, batch_grads, new_grads):
        if batch_grads is None:
            return list(new_grads)
        return [batch_grads[i] + new_grads[i] for i in range(len(new_grads))]

    def average_batch_gradients(self, batch_grads, batch_size):
        return [g / batch_size for g in batch_grads]

    # ── Master training function ──────────────────────────────────────────────

    def train(self, config):
        """
        Full training loop with mini-batches and epochs.

        config keys:
          nodes          : layer sizes e.g. (784, 256, 128, 10)
          train_file     : path to MNIST-format CSV
          images         : number of samples to train on
          epochs         : full passes over the dataset
          batch_size     : samples per gradient update (mini-batch SGD)
          learning_rate  : starting LR (decays each epoch)
          momentum       : SGD momentum coefficient (default 0.9)
          decay_rate     : LR multiplier per epoch (e.g. 0.95)
          verbose        : print progress every N samples (0 = epoch summary only)
        """
        global Nodes, Iter_Index, Hiddenlayers, WandB
        global Prevvalues, Altvalues, NWandB, OptWandB, LayerWandB
        global ACTIV, DERIV

        # ── Unpack config ──────────────────────────────────────────────────
        Nodes        = config['nodes']
        images       = config['images']
        epochs       = config['epochs']
        batch_size   = config['batch_size']
        base_lr      = config['learning_rate']
        mu           = config['momentum']
        decay_rate   = config['decay_rate']
        verbose      = config.get('verbose', 500)

        Iter_Index   = len(Nodes) - 1
        Hiddenlayers = len(Nodes) - 2

        # ── Initialize ────────────────────────────────────────────────────
        Prevvalues = self.init_var();  Altvalues  = self.init_var()
        NWandB     = self.init_var();  OptWandB   = self.init_var()
        LayerWandB = self.init_var()
        ACTIV      = self.init_hiddenlayers()
        DERIV      = self.init_hiddenlayers()
        WandB      = self.weightsandbiases()

        # ── Load data ─────────────────────────────────────────────────────
        print("Loading data...")
        X, y = self.load_data(config['train_file'])
        X, y = X[:images], y[:images]

        print(f"\nArchitecture : {' → '.join(str(n) for n in Nodes)}")
        print(f"Samples      : {images}  |  Epochs: {epochs}  |  Batch size: {batch_size}")
        print(f"Learning rate: {base_lr}  |  Momentum: {mu}  |  LR decay: {decay_rate}\n")

        global_step = 0

        for epoch in range(epochs):
            lr = self.get_lr(base_lr, epoch, decay_rate)
            correct, total_loss = 0, 0.0
            batch_grads, batch_count = None, 0

            # Reshuffle data every epoch for better generalization
            perm = np.random.permutation(images)
            X, y = X[perm], y[perm]

            for i in range(images):
                label = y[i]

                # Forward
                self.forward(X[i])
                loss = self.C_CrossEntropyLoss(label)
                if np.isnan(loss):
                    print(f"\n  !! NaN loss detected at sample {i} — stopping early.")
                    print(f"  !! Try lowering learning_rate further.")
                    accuracy = correct / max(i, 1) * 100
                    print(f"  !! Accuracy so far: {accuracy:.2f}%\n")
                    return WandB
                total_loss += loss
                correct    += self.Score(label)

                # Backward — accumulate gradients over the batch
                grads = self.Backward_Prop()
                batch_grads = self.accumulate_batch_gradients(batch_grads, grads)
                batch_count += 1

                # Update weights at end of each mini-batch (or final sample)
                if batch_count == batch_size or i == images - 1:
                    NWandB[:]     = self.average_batch_gradients(batch_grads, batch_count)
                    Prevvalues[:] = list(Altvalues)
                    WandB[:], Altvalues[:] = self.GradientDescentWithMomentum(
                        mu, lr, is_first=(global_step < batch_size)
                    )
                    batch_grads, batch_count = None, 0

                global_step += 1

                if verbose and (i + 1) % verbose == 0:
                    print(f"  Epoch {epoch+1}/{epochs} | "
                          f"Sample {i+1}/{images} | "
                          f"Loss: {total_loss/(i+1):.4f} | "
                          f"Acc: {correct/(i+1)*100:.2f}% | "
                          f"LR: {lr:.5f}")

            print(f"\n{'='*55}")
            print(f"  Epoch {epoch+1}/{epochs} COMPLETE")
            print(f"  Accuracy : {correct/images*100:.2f}%")
            print(f"  Avg Loss : {total_loss/images:.4f}")
            print(f"  LR used  : {lr:.5f}")
            print(f"{'='*55}\n")

        print("Training complete.")
        return WandB  # return final weights for saving/reuse if needed

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, test_file):
        """
        Runs inference on a labelled test CSV using trained weights.
        No weight updates — pure forward pass only.
        Reports overall accuracy and per-digit breakdown.
        Expected format: label in col 0, pixels in cols 1-784.
        """
        if WandB is None:
            raise RuntimeError("No trained weights found. Run train() before evaluate().")

        print(f"\nLoading test data from '{test_file}'...")
        data = pd.read_csv(test_file)
        arr = np.array(data)
        X_test = arr[:, 1:] / 255.0
        y_test = arr[:, 0].astype(int)
        n = len(X_test)

        correct = 0
        class_correct = np.zeros(Nodes[-1], dtype=int)
        class_total   = np.zeros(Nodes[-1], dtype=int)
        predictions   = []

        print(f"Running inference on {n} samples...\n")
        for i in range(n):
            self.forward(X_test[i])
            pred = int(np.argmax(self.softmaxlist))
            predictions.append(pred)
            label = y_test[i]
            class_total[label] += 1
            if pred == label:
                correct += 1
                class_correct[label] += 1

        test_acc = correct / n * 100
        print(f"{'='*55}")
        print(f"  TEST RESULTS")
        print(f"{'='*55}")
        print(f"  Test Accuracy : {test_acc:.2f}%  ({correct}/{n})")
        print(f"\n  Per-digit accuracy:")
        for c in range(Nodes[-1]):
            if class_total[c] > 0:
                pct = class_correct[c] / class_total[c] * 100
                bar = '█' * int(pct // 5) + '░' * (20 - int(pct // 5))
                print(f"    Digit {c}:  {bar}  {pct:.1f}%  ({class_correct[c]}/{class_total[c]})")
        print(f"{'='*55}\n")
        return test_acc, predictions


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURE AND RUN  ── edit only this block to tune your model
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    nn = Sequential()

    # ── Step 1: Train ──────────────────────────────────────────────
    nn.train({

        # Architecture: first=784 (input), last=10 (classes), middle=hidden layers
        #   3 layers  →  (784, 256, 10)
        #   4 layers  →  (784, 256, 128, 10)       ← recommended
        #   5 layers  →  (784, 256, 128, 64, 10)
        'nodes'         : (784, 256, 128, 10),

        'train_file'    : './train.csv',   # path to training CSV
        'images'        : 35700,           # samples to train on (max ~42000)

        'epochs'        : 5,               # full passes over the dataset
        'batch_size'    : 32,              # samples per weight update

        'learning_rate' : 0.01,            # restored now that D_LeakyReLU input is correct
        'momentum'      : 0.9,             # SGD momentum
        'decay_rate'    : 0.99,            # slow decay — stay near base LR longer

        'verbose'       : 500,             # print every N samples (0 = off)
    })

    # ── Step 2: Evaluate on test set ───────────────────────────────
    # Runs forward-pass only — no weight updates.
    # Reports test accuracy + per-digit breakdown to diagnose overfitting.
    nn.evaluate('./test.csv')