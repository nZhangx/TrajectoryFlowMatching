import torch

PE_BASE = 0.012 # 0.012615662610100801
NUM_FREQS = 10

def positional_encoding_tensor(time_tensor, num_frequencies=NUM_FREQS, base=PE_BASE):
    # Ensure the time tensor is in the range [0, 1]
    time_tensor = time_tensor.clamp(0, 1).unsqueeze(1)  # Clamp and add dimension for broadcasting

    # Compute the arguments for the sine and cosine functions using the custom base
    frequencies = torch.pow(base, -torch.arange(0, num_frequencies, dtype=torch.float32) / num_frequencies).to(time_tensor.device)
    angles = time_tensor * frequencies

    # Compute the sine and cosine for even and odd indices respectively
    sine = torch.sin(angles)
    cosine = torch.cos(angles)

    # Stack them along the last dimension
    pos_encoding = torch.stack((sine, cosine), dim=-1)
    pos_encoding = pos_encoding.flatten(start_dim=2)

    # Normalize to have values between 0 and 1
    pos_encoding = (pos_encoding + 1) / 2  # Now values are between 0 and 1
    
    return pos_encoding

def positional_encoding_df(df, col_mod = "time_normalized"):
    pe_tensors = torch.tensor(df[col_mod].values).astype(torch.float32)
    pos_encoding = positional_encoding_tensor(pe_tensors)
    pos_encoding_array = pos_encoding.numpy().reshape(-1, NUM_FREQS * 2)
    return pos_encoding_array
