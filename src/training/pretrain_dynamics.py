import os

from models import ReplayBuffer, GaussianMLPDynamics


CONSTRUCTORS = dict(
    gaussian_mlp=GaussianMLPDynamics
)
MODEL_ARGUMENTS = dict(
    ugv=dict(
       state_size=1,
       action_size=2,
       output_size=3 
    )
)


def find_best_model(base_dir):
    """
    Finds the subdirectory with the lowest validation loss.
    
    :param base_dir: Path to the base directory containing model subdirectories.
    :return: Path to the subdirectory with the lowest validation loss, or None if not found.
    """
    best_loss = float('inf')
    best_dir = None

    # Check if base directory exists
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory {base_dir} does not exist.")
    
    # Iterate through the subdirectories
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        
        # Ensure it's a directory
        if not os.path.isdir(subdir_path):
            continue
        
        try:
            # Split on '=' and extract validation loss
            parts = subdir.split('=')
            if len(parts) == 2 and parts[0].startswith("epoch") and "valloss" in parts[0]:
                val_loss = float(parts[1])
                
                # Update if a lower loss is found
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_dir = subdir_path
        except (ValueError, IndexError):
            # Skip invalid entries
            continue
    
    return best_dir


def load_data(train_data_paths, val_data_paths):

    train_replay_buffer = ReplayBuffer(capacity=2_000_000)
    for data_path in train_data_paths:
        train_replay_buffer.load_csv(data_path)
        print('loaded', data_path)
    print('Loaded train replay buffer:', len(train_replay_buffer), '| Capacity:', train_replay_buffer.capacity)

    val_replay_buffer = ReplayBuffer(capacity=1_000_000)
    for data_path in val_data_paths:
        val_replay_buffer.load_csv(data_path)
    print('Loaded validation replay buffer:', len(val_replay_buffer), '| Capacity:', val_replay_buffer.capacity)

    return train_replay_buffer, val_replay_buffer


def pretrain(constructor, constructor_arguments, train_data_paths, val_data_paths, save_path):

    # ####### Load Data
    train_replay_buffer, val_replay_buffer = load_data(train_data_paths, val_data_paths)

    # ####### Load Model and Train
    model = constructor(
        state_size=constructor_arguments['state_size'],
        action_size=constructor_arguments['action_size'],
        output_size=constructor_arguments['output_size'],
        device='cuda'
    )
    model.learn(
        train_dataset=train_replay_buffer,
        val_dataset=val_replay_buffer,
        persistence_path=save_path
    )


def fine_tune_model(constructor, constructor_arguments, train_data_paths, val_data_paths, artifact_path, save_path):

    # ####### Load Data
    train_replay_buffer, val_replay_buffer = load_data(train_data_paths, val_data_paths)

    # ####### Load Model and Train
    model = constructor(
        state_size=constructor_arguments['state_size'],
        action_size=constructor_arguments['action_size'],
        output_size=constructor_arguments['output_size'],
        device='cuda'
    )
    model.load(persistence_dir=artifact_path)
    model.learn(
        train_dataset=train_replay_buffer,
        val_dataset=val_replay_buffer,
        persistence_path=save_path,
        max_epochs=16
    )


def run_pretrain(platform, domrandpercent=None):
    assert platform in ['ugv']

    save_path = f'./artifacts2/{platform}/'
    base_path = f'./data/{platform}/'
    name_for_domrand = f'_domrand_{domrandpercent}perc' if domrandpercent is not None else ''
    train_data_paths = [
        os.path.join(base_path, f'{platform}_transition_dynamics{name_for_domrand}_delta.csv') #_random.csv')
    ]
    val_data_paths = [
        os.path.join(base_path, f'{platform}_transition_dynamics{name_for_domrand}_val_delta.csv')
    ]
        
    save_path = os.path.join(save_path, 'pe{}'.format(f'_domrand{domrandpercent}perc' if domrandpercent is not None and platform == 'ugv' else ''))
    for i in rng: # range(4, 5):
        print('*** Training model', i)
        print('Training data', train_data_paths)
        print('Validation data', val_data_paths)
        print('Saving to', save_path)
        pretrain(
            constructor=CONSTRUCTORS['gaussian_mlp'],
            constructor_arguments=MODEL_ARGUMENTS[platform],
            train_data_paths=train_data_paths, 
            val_data_paths=val_data_paths, 
            save_path=os.path.join(save_path, f'model_{i}')
        )


if __name__ == '__main__':

    rng = range(0, 5)
    run_pretrain(platform='ugv')
