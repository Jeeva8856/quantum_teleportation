import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.visualization import plot_histogram

def run_quantum_teleportation(theta, phi):
    qr = QuantumRegister(3, name="q")
    cr_z = ClassicalRegister(1, name="cr_z")      
    cr_x = ClassicalRegister(1, name="cr_x")      
    cr_bob = ClassicalRegister(1, name="cr_bob")
    
    qc = QuantumCircuit(qr, cr_z, cr_x, cr_bob)

    qc.u(theta, phi, 0, qr[0])
    qc.barrier(label="Input |psi>")

    qc.h(qr[1])
    qc.cx(qr[1], qr[2])
    qc.barrier(label="Bell Pair")

    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.barrier(label="Alice Ops")

    qc.measure(qr[0], cr_z[0])
    qc.measure(qr[1], cr_x[0])
    qc.barrier(label="Alice Meas")

    with qc.if_test((cr_x[0], 1)):
        qc.x(qr[2])
    with qc.if_test((cr_z[0], 1)):
        qc.z(qr[2])
    qc.barrier(label="Bob Correct")

    qc.u(theta, phi, 0, qr[2]).inverse()
    qc.measure(qr[2], cr_bob[0])

    return qc

def build_noise_model(p_1qubit=0.01, p_2qubit=0.04):
    noise_model = NoiseModel()
    err_1 = depolarizing_error(p_1qubit, 1)
    noise_model.add_all_qubit_quantum_error(err_1, ['u', 'h', 'x', 'z'])
    err_2 = depolarizing_error(p_2qubit, 2)
    noise_model.add_all_qubit_quantum_error(err_2, ['cx'])
    return noise_model

if __name__ == "__main__":
    theta_input = np.pi / 3
    phi_input = np.pi / 4

    circuit = run_quantum_teleportation(theta_input, phi_input)

    shots = 4096

    sim_ideal = AerSimulator()
    counts_ideal = sim_ideal.run(circuit, shots=shots).result().get_counts()
    bob_zeros_ideal = sum(c for k, c in counts_ideal.items() if (k.split()[0]  if ' ' in k else k[0])== '0')
    fidelity_ideal = (bob_zeros_ideal / shots) * 100

    noise_model = build_noise_model(p_1qubit=0.01, p_2qubit=0.04)
    sim_noisy = AerSimulator(noise_model=noise_model)
    counts_noisy = sim_noisy.run(circuit, shots=shots).result().get_counts()
    bob_zeros_noisy = sum(c for k, c in counts_noisy.items() if (k.split()[0]  if ' ' in k else k[0])== '0')
    fidelity_noisy = (bob_zeros_noisy / shots) * 100

    print(f"Ideal Fidelity: {fidelity_ideal:.2f}%")
    print(f"Noisy Fidelity: {fidelity_noisy:.2f}%")
    print(f"Classical Limit: 66.67%")

    fig, ax = plt.subplots(figsize=(8,4))
    plot_histogram([counts_ideal, counts_noisy], legend=['Ideal', 'Noisy(NISQ)'], ax=ax)
    fig.savefig("comparison_histogram.png", bbox_inches="tight")
    print("Saved 'comparison_histogram.png'.")


    try:
        fig_2 = circuit.draw(output="mpl")
        fig_2.savefig("circuit_diagram.png", bbox_inches="tight")
    except Exception:
        with open("circuit_diagram.txt", "w") as f:
            f.write(circuit.draw(output="text"))
