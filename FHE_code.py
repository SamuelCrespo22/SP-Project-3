import tenseal as ts
import time
import os

# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================
# Scenario: Calculating Total Portfolio Value = Sum(Shares * Prices)
# Shares (Integers)
shares_list = [10, 50, 20, 5, 100] 
# Prices (Floats)
prices_list = [150.50, 200.00, 50.25, 1000.00, 10.10]

print(f"Dataset Shares: {shares_list}")
print(f"Dataset Prices: {prices_list}")
print("--------------------------------------------------\n")

# Helper to clean up files
def cleanup():
    for f in ["enc_ckks.dat", "res_ckks.dat", "enc_bfv.dat", "res_bfv.dat"]:
        if os.path.exists(f): os.remove(f)

cleanup()

# ==============================================================================
# SCHEME 1: CKKS (Approximate Float Arithmetic)
# ==============================================================================
print("=== SCHEME 1: CKKS (Real Numbers) ===")

def setup_ckks():
    # CKKS Context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS, 
        poly_modulus_degree=8192, 
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

def holder_encrypt_ckks(context, s, p, filename):
    enc_s = ts.ckks_vector(context, s)
    enc_p = ts.ckks_vector(context, p)
    # Save combined bytes for simplicity
    with open(filename, "wb") as f:
        f.write(enc_s.serialize() + b":::" + enc_p.serialize())

def analyzer_process_ckks(context, in_file, out_file):
    with open(in_file, "rb") as f:
        data = f.read()
        bytes_s, bytes_p = data.split(b":::")
    
    enc_s = ts.ckks_vector_from(context, bytes_s)
    enc_p = ts.ckks_vector_from(context, bytes_p)
    
    # Homomorphic Multiplication and Addition (Dot Product)
    enc_result = enc_s.dot(enc_p)
    
    with open(out_file, "wb") as f:
        f.write(enc_result.serialize())

def holder_decrypt_ckks(context, filename):
    with open(filename, "rb") as f:
        bytes_res = f.read()
    enc_res = ts.ckks_vector_from(context, bytes_res)
    return enc_res.decrypt()[0]

# --- Execution CKKS ---
ctx_ckks = setup_ckks()
start = time.time()

holder_encrypt_ckks(ctx_ckks, shares_list, prices_list, "enc_ckks.dat")
t_enc = time.time() - start

start_proc = time.time()
analyzer_process_ckks(ctx_ckks, "enc_ckks.dat", "res_ckks.dat")
t_proc = time.time() - start_proc

start_dec = time.time()
final_ckks = holder_decrypt_ckks(ctx_ckks, "res_ckks.dat")
t_dec = time.time() - start_dec

print(f"CKKS Total Value: {final_ckks:.4f}")
print(f"Times: Enc={t_enc:.4f}s | Proc={t_proc:.4f}s | Dec={t_dec:.4f}s\n")


# ==============================================================================
# SCHEME 2: BFV (Exact Integer Arithmetic)
# ==============================================================================
print("=== SCHEME 2: BFV (Integers) ===")
# Note: BFV only supports integers. We must scale our float prices to ints (cents).

def setup_bfv():
    # BFV Context
    # FIX: Increased poly_modulus_degree and plain_modulus to prevent overflow
    context = ts.context(
        ts.SCHEME_TYPE.BFV, 
        poly_modulus_degree=8192, 
        plain_modulus=536903681 # Large prime to accommodate the sum of ~1.8 million
    )
    context.generate_galois_keys()
    context.generate_relin_keys() # Required for multiplication
    return context

def holder_encrypt_bfv(context, s, p, filename):
    # Scale prices (x100) to make them integers
    p_integers = [int(val * 100) for val in p]
    s_integers = [int(val) for val in s] # Ensure shares are ints
    
    enc_s = ts.bfv_vector(context, s_integers)
    enc_p = ts.bfv_vector(context, p_integers)
    
    with open(filename, "wb") as f:
        f.write(enc_s.serialize() + b":::" + enc_p.serialize())

def analyzer_process_bfv(context, in_file, out_file):
    with open(in_file, "rb") as f:
        data = f.read()
        bytes_s, bytes_p = data.split(b":::")
    
    enc_s = ts.bfv_vector_from(context, bytes_s)
    enc_p = ts.bfv_vector_from(context, bytes_p)
    
    # Homomorphic Dot Product
    enc_result = enc_s.dot(enc_p)
    
    with open(out_file, "wb") as f:
        f.write(enc_result.serialize())

def holder_decrypt_bfv(context, filename):
    with open(filename, "rb") as f:
        bytes_res = f.read()
    enc_res = ts.bfv_vector_from(context, bytes_res)
    
    # Decrypt gives us the total in "cents"
    total_cents = enc_res.decrypt()[0]
    
    # Scale back to dollars
    return total_cents / 100.0

# --- Execution BFV ---
ctx_bfv = setup_bfv()
start = time.time()

holder_encrypt_bfv(ctx_bfv, shares_list, prices_list, "enc_bfv.dat")
t_enc_bfv = time.time() - start

start_proc = time.time()
analyzer_process_bfv(ctx_bfv, "enc_bfv.dat", "res_bfv.dat")
t_proc_bfv = time.time() - start_proc

start_dec = time.time()
final_bfv = holder_decrypt_bfv(ctx_bfv, "res_bfv.dat")
t_dec_bfv = time.time() - start_dec

print(f"BFV Total Value:  {final_bfv:.4f}")
print(f"Times: Enc={t_enc_bfv:.4f}s | Proc={t_proc_bfv:.4f}s | Dec={t_dec_bfv:.4f}s\n")

# ==============================================================================
# VERIFICATION
# ==============================================================================
real_total = sum([s*p for s, p in zip(shares_list, prices_list)])
print(f"Ground Truth:     {real_total:.4f}")
print(f"CKKS Error:       {abs(final_ckks - real_total):.6f}")
print(f"BFV Error:        {abs(final_bfv - real_total):.6f}")