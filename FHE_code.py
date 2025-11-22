import tenseal as ts
from phe import paillier
import time


# Input Data
salaries = [1200.50, 2500.00, 1800.75, 3200.10, 4500.00]
print(f"Original Data: {salaries}\n")

# ==============================================================================
# CKKS Scheme (Using tenseal)
# ==============================================================================
print("__________________________________________________")
print("___ Starting Execution: CKKS (TENSEAL).        ___")
print("__________________________________________________")

# Key Generation
def setup_ckks():
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

# Data Holder
# Encrypt
def holder_encrypt_ckks(context, data):
    enc_vector = ts.ckks_vector(context, data)
    return enc_vector.serialize()

# Decrypt
def holder_decrypt_ckks(context, encrypted_result):
    enc_result = ts.ckks_vector_from(context, encrypted_result)
    return enc_result.decrypt()[0]

# Data Analyzer
def analyzer_process_ckks(context, encrypted_data):
    enc_vector = ts.ckks_vector_from(context, encrypted_data)
    
    enc_sum = enc_vector.sum()
    enc_avg = enc_sum * (1.0 / len(salaries))
    
    return enc_avg.serialize()


# Execution (CKKS)
ctx = setup_ckks()
start_ckks = time.time()

# 1 - Encryption
encrypted_data_ckks = holder_encrypt_ckks(ctx, salaries)
t_enc_ckks = time.time() - start_ckks

# 2 - Processing
start_proc = time.time()
encrypted_result_ckks = analyzer_process_ckks(ctx, encrypted_data_ckks)
t_proc_ckks = time.time() - start_proc

# 3 - Decryption
start_dec = time.time()
result_ckks = holder_decrypt_ckks(ctx, encrypted_result_ckks)
t_dec_ckks = time.time() - start_dec

print(f"-> Result (Average): {result_ckks:.4f}")
print(f"-> Times: Enc={t_enc_ckks:.4f}s | Proc={t_proc_ckks:.4f}s | Dec={t_dec_ckks:.4f}s\n")


# ==============================================================================
# Paillier Scheme (PHE)
# ==============================================================================
print("__________________________________________________")
print("___ Starting Execution: Paillier (PHE).        ___")
print("__________________________________________________")

# Generate Keys
def setup_pail():
    public_key, private_key = paillier.generate_paillier_keypair()
    return public_key, private_key

# Data Holder
# Encryption
def holder_encrypt_pail(public_key, data):
    return [public_key.encrypt(x) for x in data]

# Decryption
def holder_decrypt_pail(private_key, enc_result):
    return private_key.decrypt(enc_result)

# Data Analyzer
def analyzer_process_pail(enc_list):
    enc_sum = sum(enc_list[1:], enc_list[0])
    enc_avg = enc_sum / len(enc_list)
    return enc_avg

# Execution (Paillier)
pub_k, priv_k = setup_pail()
start_pail = time.time()

# Encryption
encrypted_data_pail = holder_encrypt_pail(pub_k, salaries)
t_enc_pail = time.time() - start_pail

# Processing
start_proc = time.time()
encrypted_result_pail = analyzer_process_pail(encrypted_data_pail)
t_proc_pail = time.time() - start_proc

# Decryption
start_dec = time.time()
result_pail = holder_decrypt_pail(priv_k, encrypted_result_pail)
t_dec_pail = time.time() - start_dec

print(f"-> Result (Average): {result_pail:.4f}")
print(f"-> Times: Enc={t_enc_pail:.4f}s | Proc={t_proc_pail:.4f}s | Dec={t_dec_pail:.4f}s\n")

# ==========================================
# Final Comparison
# ==========================================
print("__________________________________________________")
print("___ Result Comparison.                         ___")
print("__________________________________________________")
real_avg = sum(salaries) / len(salaries)
print(f"Real Average:     {real_avg:.4f}")
print(f"CKKS Average:  {result_ckks:.4f}")
print(f"Paillier Average: {result_pail:.4f}")
print(f"\nCKKS Error:  {abs(result_ckks - real_avg):.6f}")
print(f"Paillier Error: {abs(result_pail - real_avg):.6f}")