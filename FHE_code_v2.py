import tenseal as ts
import pandas as pd
import time
import os
import math

# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================

df = pd.read_csv("datasets/dataset.csv")
salaries_list = df["salary_cents"].tolist()
bonus_list = df["bonus_cents"].tolist()

print(f"Dataset Size: {len(salaries_list)} rows")
print("=" * 80)
print("\n")

def cleanup():
    for f in ["enc_ckks.dat", "res_ckks_stats.dat", "enc_bfv.dat", "res_bfv_stats.dat",
              "enc_ckks_perf.dat", "res_ckks_add.dat", "res_ckks_mul.dat",
              "enc_bfv_perf.dat", "res_bfv_add.dat", "res_bfv_mul.dat"]:
        if os.path.exists(f): os.remove(f)

cleanup()

# ==============================================================================
# SCHEME 1: CKKS - STATISTICS (Mean, Variance, Z-Score)
# ==============================================================================

print("=" * 80)
print("CKKS - STATISTICAL ANALYSIS")
print("=" * 80)

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
    
    # Calculate statistics for salaries
    enc_salary_mean = enc_s.sum() * (1.0 / len(salaries_list))
    enc_salary_variance = ((enc_s - enc_salary_mean.decrypt()[0]) ** 2).sum() * (1.0 / len(salaries_list))
    
    # Calculate statistics for bonuses
    enc_bonus_mean = enc_b.sum() * (1.0 / len(bonus_list))
    enc_bonus_variance = ((enc_b - enc_bonus_mean.decrypt()[0]) ** 2).sum() * (1.0 / len(bonus_list))
        
    # Calculate Result = (salary + 0.1 * bonus) * 1.05
    enc_result = (enc_s + enc_b.mul(0.1)).mul(1.05)
    enc_total = enc_result.sum()
    
    # Serialize all results
    with open(out_file, "wb") as f:
        ser_sal_mean = enc_salary_mean.serialize()
        ser_sal_var = enc_salary_variance.serialize()
        ser_bon_mean = enc_bonus_mean.serialize()
        ser_bon_var = enc_bonus_variance.serialize()
        ser_total = enc_total.serialize()
        
        f.write(len(ser_sal_mean).to_bytes(4, 'big'))
        f.write(ser_sal_mean)
        f.write(len(ser_sal_var).to_bytes(4, 'big'))
        f.write(ser_sal_var)
        f.write(len(ser_bon_mean).to_bytes(4, 'big'))
        f.write(ser_bon_mean)
        f.write(len(ser_bon_var).to_bytes(4, 'big'))
        f.write(ser_bon_var)
        f.write(len(ser_total).to_bytes(4, 'big'))
        f.write(ser_total)

def holder_decrypt_ckks(context, filename):
    with open(filename, "rb") as f:
        # Read salary mean
        size1 = int.from_bytes(f.read(4), 'big')
        enc_sal_mean = ts.ckks_vector_from(context, f.read(size1))
        
        # Read salary variance
        size2 = int.from_bytes(f.read(4), 'big')
        enc_sal_var = ts.ckks_vector_from(context, f.read(size2))
        
        # Read bonus mean
        size3 = int.from_bytes(f.read(4), 'big')
        enc_bon_mean = ts.ckks_vector_from(context, f.read(size3))
        
        # Read bonus variance
        size4 = int.from_bytes(f.read(4), 'big')
        enc_bon_var = ts.ckks_vector_from(context, f.read(size4))
        
        # Read total result
        size5 = int.from_bytes(f.read(4), 'big')
        enc_total = ts.ckks_vector_from(context, f.read(size5))
    
    salary_mean = enc_sal_mean.decrypt()[0]
    salary_var = enc_sal_var.decrypt()[0]
    bonus_mean = enc_bon_mean.decrypt()[0]
    bonus_var = enc_bon_var.decrypt()[0]
    total_result = enc_total.decrypt()[0]
    
    # Calculate standard deviations
    salary_std = math.sqrt(salary_var)
    bonus_std = math.sqrt(bonus_var)
    
    return {
        'salary_mean': salary_mean,
        'salary_variance': salary_var,
        'salary_std': salary_std,
        'bonus_mean': bonus_mean,
        'bonus_variance': bonus_var,
        'bonus_std': bonus_std,
        'total_result': total_result
    }

# Execution CKKS Statistics
ctx_ckks = setup_ckks()
start_total = time.time()

start = time.time()
holder_encrypt_ckks(ctx_ckks, salaries_list, bonus_list, "enc_ckks.dat")
t_enc = time.time() - start

start_proc = time.time()
analyzer_process_ckks(ctx_ckks, "enc_ckks.dat", "res_ckks_stats.dat")
t_proc = time.time() - start_proc

start_dec = time.time()
ckks_stats = holder_decrypt_ckks(ctx_ckks, "res_ckks_stats.dat")
t_dec = time.time() - start_dec

t_total_ckks = time.time() - start_total

print("\n--- CKKS RESULTS ---")
print(f"Salary Mean:      {ckks_stats['salary_mean']:.2f}")
print(f"Salary Variance:  {ckks_stats['salary_variance']:.2f}")
print(f"Salary Std Dev:   {ckks_stats['salary_std']:.2f}")
print(f"\nBonus Mean:       {ckks_stats['bonus_mean']:.2f}")
print(f"Bonus Variance:   {ckks_stats['bonus_variance']:.2f}")
print(f"Bonus Std Dev:    {ckks_stats['bonus_std']:.2f}")

# Z-Score examples (for first value in each list)
z_score_salary_first = (salaries_list[0] - ckks_stats['salary_mean']) / ckks_stats['salary_std']
z_score_bonus_first = (bonus_list[0] - ckks_stats['bonus_mean']) / ckks_stats['bonus_std']
print(f"\nZ-Score (First Salary): {z_score_salary_first:.4f}")
print(f"Z-Score (First Bonus):  {z_score_bonus_first:.4f}")

print(f"\nTotal Result [(salary + 0.1*bonus) * 1.05]: {ckks_stats['total_result']:.2f}")

print(f"\n--- Timing ---")
print(f"Encryption:  {t_enc:.4f}s")
print(f"Processing:  {t_proc:.4f}s")
print(f"Decryption:  {t_dec:.4f}s")
print(f"Total Time:  {t_total_ckks:.4f}s")
print("\n")

# ==============================================================================
# SCHEME 2: BFV - STATISTICS (Sum)
# ==============================================================================

print("=" * 80)
print("BFV - STATISTICAL ANALYSIS")
print("=" * 80)

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
    
    # Calculate sum for salaries and bonuses
    enc_salary_sum = enc_s.sum()
    enc_bonus_sum = enc_b.sum()
    
    # Calculate Result = (salary + 0.1 * bonus) * 1.05
    # Using integer arithmetic: (10*S + B) * 21 / 200
    enc_result = (enc_s.mul(10) + enc_b).mul(21).sum()
    
    # Serialize results
    with open(out_file, "wb") as f:
        ser_sal_sum = enc_salary_sum.serialize()
        ser_bon_sum = enc_bonus_sum.serialize()
        ser_total = enc_result.serialize()
        
        f.write(len(ser_sal_sum).to_bytes(4, 'big'))
        f.write(ser_sal_sum)
        f.write(len(ser_bon_sum).to_bytes(4, 'big'))
        f.write(ser_bon_sum)
        f.write(len(ser_total).to_bytes(4, 'big'))
        f.write(ser_total)

def holder_decrypt_bfv(context, filename):
    with open(filename, "rb") as f:
        # Read salary sum
        size1 = int.from_bytes(f.read(4), 'big')
        enc_sal_sum = ts.bfv_vector_from(context, f.read(size1))
        
        # Read bonus sum
        size2 = int.from_bytes(f.read(4), 'big')
        enc_bon_sum = ts.bfv_vector_from(context, f.read(size2))
        
        # Read total result
        size3 = int.from_bytes(f.read(4), 'big')
        enc_total = ts.bfv_vector_from(context, f.read(size3))
    
    salary_sum = enc_sal_sum.decrypt()[0]
    bonus_sum = enc_bon_sum.decrypt()[0]
    total_numerator = enc_total.decrypt()[0]
    
    # Apply the division by 200 (from the 10 * 20 scaling factor)
    total_result = total_numerator / 200.0
    
    return {
        'salary_sum': salary_sum,
        'bonus_sum': bonus_sum,
        'total_result': total_result
    }

# Execution BFV Statistics
ctx_bfv = setup_bfv()
start_total = time.time()

start = time.time()
holder_encrypt_bfv(ctx_bfv, salaries_list, bonus_list, "enc_bfv.dat")
t_enc_bfv = time.time() - start

start_proc = time.time()
analyzer_process_bfv(ctx_bfv, "enc_bfv.dat", "res_bfv_stats.dat")
t_proc_bfv = time.time() - start_proc

start_dec = time.time()
bfv_stats = holder_decrypt_bfv(ctx_bfv, "res_bfv_stats.dat")
t_dec_bfv = time.time() - start_dec

t_total_bfv = time.time() - start_total

print("\n--- BFV RESULTS ---")
print(f"Salary Sum:  {bfv_stats['salary_sum']}")
print(f"Bonus Sum:   {bfv_stats['bonus_sum']}")
print(f"\nTotal Result [(salary + 0.1*bonus) * 1.05]: {bfv_stats['total_result']:.2f}")

print(f"\n--- Timing ---")
print(f"Encryption:  {t_enc_bfv:.4f}s")
print(f"Processing:  {t_proc_bfv:.4f}s")
print(f"Decryption:  {t_dec_bfv:.4f}s")
print(f"Total Time:  {t_total_bfv:.4f}s")
print("\n")

# ==============================================================================
# PERFORMANCE COMPARISON - ADDITION
# ==============================================================================

print("=" * 80)
print("PERFORMANCE COMPARISON - ADDITION OF ALL SALARIES")
print("=" * 80)

# --- CKKS ADDITION ---
def analyzer_process_ckks_addition(context, in_file, out_file):
    """Addition of all salaries"""
    with open(in_file, "rb") as f:
        size_s = int.from_bytes(f.read(4), 'big')
        bytes_s = f.read(size_s)
        bytes_b = f.read()  # We still need to read bonus data
    
    enc_s = ts.ckks_vector_from(context, bytes_s)
    enc_salary_sum = enc_s.sum()
    
    with open(out_file, "wb") as f:
        f.write(enc_salary_sum.serialize())

def holder_decrypt_ckks_simple(context, filename):
    with open(filename, "rb") as f:
        bytes_res = f.read()
    enc_res = ts.ckks_vector_from(context, bytes_res)
    return enc_res.decrypt()[0]

print("\n--- CKKS Addition ---")
start_total_ckks_add = time.time()

start = time.time()
holder_encrypt_ckks(ctx_ckks, salaries_list, bonus_list, "enc_ckks_perf.dat")
t_enc_ckks_add = time.time() - start

start = time.time()
analyzer_process_ckks_addition(ctx_ckks, "enc_ckks_perf.dat", "res_ckks_add.dat")
t_proc_ckks_add = time.time() - start

start = time.time()
result_ckks_add = holder_decrypt_ckks_simple(ctx_ckks, "res_ckks_add.dat")
t_dec_ckks_add = time.time() - start

t_total_ckks_add = time.time() - start_total_ckks_add

print(f"Result: {result_ckks_add:.2f}")
print(f"Encryption:  {t_enc_ckks_add:.4f}s")
print(f"Processing:  {t_proc_ckks_add:.4f}s")
print(f"Decryption:  {t_dec_ckks_add:.4f}s")
print(f"Total Time:  {t_total_ckks_add:.4f}s")

# --- BFV ADDITION ---
def analyzer_process_bfv_addition(context, in_file, out_file):
    """Addition of all salaries"""
    with open(in_file, "rb") as f:
        size_s = int.from_bytes(f.read(4), 'big')
        bytes_s = f.read(size_s)
    
    enc_s = ts.bfv_vector_from(context, bytes_s)
    enc_salary_sum = enc_s.sum()
    
    with open(out_file, "wb") as f:
        f.write(enc_salary_sum.serialize())

def holder_decrypt_bfv_simple(context, filename):
    with open(filename, "rb") as f:
        bytes_res = f.read()
    enc_res = ts.bfv_vector_from(context, bytes_res)
    return enc_res.decrypt()[0]

print("\n--- BFV Addition ---")
start_total_bfv_add = time.time()

start = time.time()
holder_encrypt_bfv(ctx_bfv, salaries_list, bonus_list, "enc_bfv_perf.dat")
t_enc_bfv_add = time.time() - start

start = time.time()
analyzer_process_bfv_addition(ctx_bfv, "enc_bfv_perf.dat", "res_bfv_add.dat")
t_proc_bfv_add = time.time() - start

start = time.time()
result_bfv_add = holder_decrypt_bfv_simple(ctx_bfv, "res_bfv_add.dat")
t_dec_bfv_add = time.time() - start

t_total_bfv_add = time.time() - start_total_bfv_add

print(f"Result: {result_bfv_add}")
print(f"Encryption:  {t_enc_bfv_add:.4f}s")
print(f"Processing:  {t_proc_bfv_add:.4f}s")
print(f"Decryption:  {t_dec_bfv_add:.4f}s")
print(f"Total Time:  {t_total_bfv_add:.4f}s")
print("\n")

# ==============================================================================
# PERFORMANCE COMPARISON - MULTIPLICATION
# ==============================================================================

print("=" * 80)
print("PERFORMANCE COMPARISON - MULTIPLICATION OF ALL SALARIES BY 2")
print("=" * 80)

# --- CKKS MULTIPLICATION ---
def analyzer_process_ckks_multiplication(context, in_file, out_file):
    """Multiplication of all salaries by 2"""
    with open(in_file, "rb") as f:
        size_s = int.from_bytes(f.read(4), 'big')
        bytes_s = f.read(size_s)

    enc_s = ts.ckks_vector_from(context, bytes_s)
    enc_salary_mul = enc_s.mul(2)
    enc_result = enc_salary_mul.sum()
    
    with open(out_file, "wb") as f:
        f.write(enc_result.serialize())

print("\n--- CKKS Multiplication ---")
start_total_ckks_mul = time.time()

start = time.time()
holder_encrypt_ckks(ctx_ckks, salaries_list, bonus_list, "enc_ckks_perf.dat")
t_enc_ckks_mul = time.time() - start

start = time.time()
analyzer_process_ckks_multiplication(ctx_ckks, "enc_ckks_perf.dat", "res_ckks_mul.dat")
t_proc_ckks_mul = time.time() - start

start = time.time()
result_ckks_mul = holder_decrypt_ckks_simple(ctx_ckks, "res_ckks_mul.dat")
t_dec_ckks_mul = time.time() - start

t_total_ckks_mul = time.time() - start_total_ckks_mul

print(f"Result: {result_ckks_mul:.2f}")
print(f"Encryption:  {t_enc_ckks_mul:.4f}s")
print(f"Processing:  {t_proc_ckks_mul:.4f}s")
print(f"Decryption:  {t_dec_ckks_mul:.4f}s")
print(f"Total Time:  {t_total_ckks_mul:.4f}s")

# --- BFV MULTIPLICATION ---
def analyzer_process_bfv_multiplication(context, in_file, out_file):
    with open(in_file, "rb") as f:
        size_s = int.from_bytes(f.read(4), 'big')
        bytes_s = f.read(size_s)
        bytes_b = f.read()  # We still need to read bonus data
    
    enc_s = ts.bfv_vector_from(context, bytes_s)
    enc_salary_mul = enc_s.mul(2)
    enc_result = enc_salary_mul.sum()
    
    with open(out_file, "wb") as f:
        f.write(enc_result.serialize())

print("\n--- BFV Multiplication ---")
start_total_bfv_mul = time.time()

start = time.time()
holder_encrypt_bfv(ctx_bfv, salaries_list, bonus_list, "enc_bfv_perf.dat")
t_enc_bfv_mul = time.time() - start

start = time.time()
analyzer_process_bfv_multiplication(ctx_bfv, "enc_bfv_perf.dat", "res_bfv_mul.dat")
t_proc_bfv_mul = time.time() - start

start = time.time()
result_bfv_mul = holder_decrypt_bfv_simple(ctx_bfv, "res_bfv_mul.dat")
t_dec_bfv_mul = time.time() - start

t_total_bfv_mul = time.time() - start_total_bfv_mul

print(f"Result: {result_bfv_mul}")
print(f"Encryption:  {t_enc_bfv_mul:.4f}s")
print(f"Processing:  {t_proc_bfv_mul:.4f}s")
print(f"Decryption:  {t_dec_bfv_mul:.4f}s")
print(f"Total Time:  {t_total_bfv_mul:.4f}s")
print("\n")

# ==============================================================================
# VERIFICATION AND COMPARISON
# ==============================================================================

print("=" * 80)
print("VERIFICATION AND COMPARISON")
print("=" * 80)

# Ground truth calculations
real_salary_sum = sum(salaries_list)
real_bonus_sum = sum(bonus_list)
real_salary_mean = sum(salaries_list) / len(salaries_list)
real_bonus_mean = sum(bonus_list) / len(bonus_list)
real_salary_var = sum([(s - real_salary_mean)**2 for s in salaries_list]) / len(salaries_list)
real_bonus_var = sum([(b - real_bonus_mean)**2 for b in bonus_list]) / len(bonus_list)
real_salary_mul = sum([s * 2 for s in salaries_list])

print("\n--- Statistics Verification ---")
print(f"\nSalary Mean:")
print(f"  Ground Truth: {real_salary_mean:.2f}")
print(f"  CKKS Result:  {ckks_stats['salary_mean']:.2f}")
print(f"  Error:        {abs(ckks_stats['salary_mean'] - real_salary_mean):.4f}")

print(f"\nSalary Variance:")
print(f"  Ground Truth: {real_salary_var:.2f}")
print(f"  CKKS Result:  {ckks_stats['salary_variance']:.2f}")
print(f"  Error:        {abs(ckks_stats['salary_variance'] - real_salary_var):.4f}")

print(f"\nBonus Mean:")
print(f"  Ground Truth: {real_bonus_mean:.2f}")
print(f"  CKKS Result:  {ckks_stats['bonus_mean']:.2f}")
print(f"  Error:        {abs(ckks_stats['bonus_mean'] - real_bonus_mean):.4f}")

print(f"\nBonus Variance:")
print(f"  Ground Truth: {real_bonus_var:.2f}")
print(f"  CKKS Result:  {ckks_stats['bonus_variance']:.2f}")
print(f"  Error:        {abs(ckks_stats['bonus_variance'] - real_bonus_var):.4f}")

print(f"\nSalary Sum (BFV):")
print(f"  Ground Truth: {real_salary_sum}")
print(f"  BFV Result:   {bfv_stats['salary_sum']}")
print(f"  Error:        {abs(bfv_stats['salary_sum'] - real_salary_sum)}")

print(f"\nBonus Sum (BFV):")
print(f"  Ground Truth: {real_bonus_sum}")
print(f"  BFV Result:   {bfv_stats['bonus_sum']}")
print(f"  Error:        {abs(bfv_stats['bonus_sum'] - real_bonus_sum)}")

print(f"\nTotal Result [(salary + 0.1*bonus) * 1.05]:")
real_total = sum([(s + 0.1 * b) * 1.05 for s, b in zip(salaries_list, bonus_list)])
print(f"  Ground Truth: {real_total:.2f}")
print(f"  CKKS Result:  {ckks_stats['total_result']:.2f}")
print(f"  BFV Result:   {bfv_stats['total_result']:.2f}")
print(f"  CKKS Error:   {abs(ckks_stats['total_result'] - real_total):.4f}")
print(f"  BFV Error:    {abs(bfv_stats['total_result'] - real_total):.4f}")

print("\n--- Performance Verification ---")
print(f"\nAddition (Sum of Salaries):")
print(f"  Ground Truth: {real_salary_sum}")
print(f"  CKKS Result:  {result_ckks_add:.2f}")
print(f"  BFV Result:   {result_bfv_add}")
print(f"  CKKS Error:   {abs(result_ckks_add - real_salary_sum):.4f}")
print(f"  BFV Error:    {abs(result_bfv_add - real_salary_sum)}")

print(f"\nMultiplication (Salaries * 2):")
print(f"  Ground Truth: {real_salary_mul}")
print(f"  CKKS Result:  {result_ckks_mul:.2f}")
print(f"  BFV Result:   {result_bfv_mul}")
print(f"  CKKS Error:   {abs(result_ckks_mul - real_salary_mul):.4f}")
print(f"  BFV Error:    {abs(result_bfv_mul - real_salary_mul)}")

print("\n--- Performance Summary ---")
print(f"\nCKKS vs BFV - Addition:")
print(f"  CKKS Total Time: {t_total_ckks_add:.4f}s")
print(f"  BFV Total Time:  {t_total_bfv_add:.4f}s")
print(f"  Speedup:         {t_total_ckks_add / t_total_bfv_add:.2f}x {'(BFV faster)' if t_total_bfv_add < t_total_ckks_add else '(CKKS faster)'}")

print(f"\nCKKS vs BFV - Multiplication:")
print(f"  CKKS Total Time: {t_total_ckks_mul:.4f}s")
print(f"  BFV Total Time:  {t_total_bfv_mul:.4f}s")
print(f"  Speedup:         {t_total_ckks_mul / t_total_bfv_mul:.2f}x {'(BFV faster)' if t_total_bfv_mul < t_total_ckks_mul else '(CKKS faster)'}")

print("\n" + "=" * 80)