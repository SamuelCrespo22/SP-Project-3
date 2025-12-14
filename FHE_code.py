import tenseal as ts
import pandas as pd
import time
import os

# Result = (salary + 0.1* bonus) * 1.05

# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================

df = pd.read_csv("datasets/dataset.csv")
salaries_list = df["salary_cents"].tolist()
bonus_list = df["bonus_cents"].tolist()

print(f"Dataset Size: {len(salaries_list)} rows")
print("--------------------------------------------------\n")

def cleanup():
    for f in ["enc_ckks.dat", "res_ckks.dat", "enc_bfv.dat", "res_bfv.dat"]:
        if os.path.exists(f): os.remove(f)

cleanup()

# ==============================================================================
# SCHEME 1: CKKS (Approximate Float Arithmetic)
# ==============================================================================
print("=== SCHEME 1: CKKS ===")

def setup_ckks():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS, 
        poly_modulus_degree=16384, 
        coeff_mod_bit_sizes=[60, 40, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

def holder_encrypt_ckks(context, salary, bonus, filename):
    enc_s = ts.ckks_vector(context, salary)
    enc_b = ts.ckks_vector(context, bonus)

    ser_s = enc_s.serialize()
    ser_b = enc_b.serialize()
    
    with open(filename, "wb") as f:
        f.write(len(ser_s).to_bytes(4, 'big')) 
        f.write(ser_s)
        f.write(ser_b)

def analyzer_process_ckks(context, in_file, out_file):
    with open(in_file, "rb") as f:
        size_s = int.from_bytes(f.read(4), 'big')
        bytes_s = f.read(size_s)
        bytes_b = f.read()
    
    enc_s = ts.ckks_vector_from(context, bytes_s)
    enc_b = ts.ckks_vector_from(context, bytes_b)
    
    enc_result = (enc_s + enc_b.mul(0.1)).mul(1.05)
    enc_total = enc_result.sum()

    with open(out_file, "wb") as f:
        ser_total = enc_total.serialize()
        f.write(len(ser_total).to_bytes(4, 'big'))
        f.write(ser_total)

def holder_decrypt_ckks(context, filename):
    with open(filename, "rb") as f:
        size5 = int.from_bytes(f.read(4), 'big')
        enc_total = ts.ckks_vector_from(context, f.read(size5))
    total_result = enc_total.decrypt()[0]
    return total_result

# Execution CKKS
ctx_ckks = setup_ckks()
start = time.time()

holder_encrypt_ckks(ctx_ckks, salaries_list, bonus_list, "enc_ckks.dat")
t_enc = time.time() - start

start_proc = time.time()
analyzer_process_ckks(ctx_ckks, "enc_ckks.dat", "res_ckks.dat")
t_proc = time.time() - start_proc

start_dec = time.time()
final_ckks = holder_decrypt_ckks(ctx_ckks, "res_ckks.dat")
t_dec = time.time() - start_dec

print(f"CKKS Total Value: {final_ckks:.4f}")
print(f"Times: Enc={t_enc:.4f}s | Proc={t_proc:.4f}s | Dec={t_dec:.4f}s\n")
total_time_ckks = t_enc + t_proc + t_dec
print("Total Time rounded:", round(total_time_ckks, 4))
# ==============================================================================
# SCHEME 2: BFV (Exact Integer Arithmetic)
# ==============================================================================
print("=== SCHEME 2: BFV ===")

def setup_bfv():
    context = ts.context(
        ts.SCHEME_TYPE.BFV, 
        poly_modulus_degree=16384,
        plain_modulus=1099511922689, # no overflow
    )
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context

def holder_encrypt_bfv(context, salary, bonus, filename):
    enc_s = ts.bfv_vector(context, salary)
    enc_b = ts.bfv_vector(context, bonus)

    ser_s = enc_s.serialize()
    ser_b = enc_b.serialize()
    
    with open(filename, "wb") as f:
        f.write(len(ser_s).to_bytes(4, 'big'))
        f.write(ser_s)
        f.write(ser_b)

def analyzer_process_bfv(context, in_file, out_file):
    with open(in_file, "rb") as f:
        size_s = int.from_bytes(f.read(4), 'big')
        bytes_s = f.read(size_s)
        bytes_b = f.read()
    
    enc_s = ts.bfv_vector_from(context, bytes_s)
    enc_b = ts.bfv_vector_from(context, bytes_b)
    
    enc_result = (enc_s.mul(10) + enc_b).mul(21).sum()
    
    with open(out_file, "wb") as f:
        ser_total = enc_result.serialize()
        
        f.write(len(ser_total).to_bytes(4, 'big'))
        f.write(ser_total)

def holder_decrypt_bfv(context, filename):
    with open(filename, "rb") as f:
        size3 = int.from_bytes(f.read(4), 'big')
        enc_total = ts.bfv_vector_from(context, f.read(size3))
    
    total_numerator = enc_total.decrypt()[0]
    total_result = total_numerator / 200.0
    
    return total_result

# Execution BFV
ctx_bfv = setup_bfv()
start = time.time()

holder_encrypt_bfv(ctx_bfv, salaries_list, bonus_list, "enc_bfv.dat")
t_enc_bfv = time.time() - start

start_proc = time.time()
analyzer_process_bfv(ctx_bfv, "enc_bfv.dat", "res_bfv.dat")
t_proc_bfv = time.time() - start_proc

start_dec = time.time()
final_bfv = holder_decrypt_bfv(ctx_bfv, "res_bfv.dat")
t_dec_bfv = time.time() - start_dec

print(f"BFV Total Value:  {final_bfv:.4f}")
print(f"Times: Enc={t_enc_bfv:.4f}s | Proc={t_proc_bfv:.4f}s | Dec={t_dec_bfv:.4f}s\n")
total_time_bfv = t_enc_bfv + t_proc_bfv + t_dec_bfv
# Show total time rounded
print("Total Time rounded:", round(total_time_bfv, 4))

# ==============================================================================
# VERIFICATION
# ==============================================================================
# (S + 0.1*B) * 1.05
real_total = sum([(s + 0.1 * b) * 1.05 for s, b in zip(salaries_list, bonus_list)])
print(f"Ground Truth:     {real_total:.4f}")
print(f"CKKS Error:       {abs(final_ckks - real_total):.6f}")
print(f"BFV Error:        {abs(final_bfv - real_total):.6f}")