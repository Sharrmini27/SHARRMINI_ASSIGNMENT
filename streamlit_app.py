import streamlit as st
import pandas as pd
import random
import csv
import numpy as np
import os

# ============================================
# 📘 GENETIC ALGORITHM ENGINE
# ============================================

def read_csv_to_dict(file_path):
    """Reads program ratings CSV and returns dictionary {Program: [ratings]}"""
    program_ratings = {}
    try:
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader, None)
            for row in reader:
                if len(row) > 1:
                    program = row[0]
                    ratings = []
                    for x in row[1:]:
                        try:
                            ratings.append(float(x))
                        except ValueError:
                            st.warning(f"Invalid rating in {program}, skipped value: {x}")
                    program_ratings[program] = ratings
    except FileNotFoundError:
        st.error(f"Error: File '{file_path}' not found.")
    return program_ratings


def fitness_function(schedule, ratings_data):
    """Calculates total fitness for a schedule."""
    return sum(
        ratings_data.get(program, [0])[i % len(ratings_data.get(program, [0]))]
        for i, program in enumerate(schedule)
    )


def create_random_schedule(all_programs, schedule_length):
    return [random.choice(all_programs) for _ in range(schedule_length)]


def crossover(schedule1, schedule2):
    """Single-point crossover."""
    if len(schedule1) < 2 or len(schedule2) < 2:
        return schedule1, schedule2
    point = random.randint(1, len(schedule1) - 1)
    child1 = schedule1[:point] + schedule2[point:]
    child2 = schedule2[:point] + schedule1[point:]
    return child1, child2


def mutate(schedule, all_programs, mutation_strength=0.1):
    """Mutates one or more genes in a schedule."""
    schedule_copy = schedule.copy()
    for _ in range(random.randint(1, 3)):  # mutate up to 3 genes
        mutation_point = random.randint(0, len(schedule_copy) - 1)
        schedule_copy[mutation_point] = random.choice(all_programs)
    return schedule_copy


def genetic_algorithm(ratings_data, all_programs, schedule_length,
                      generations=100, population_size=50,
                      crossover_rate=0.8, mutation_rate=0.2, elitism_size=2):
    """Core GA loop."""
    population = [create_random_schedule(all_programs, schedule_length) for _ in range(population_size)]
    best_schedule, best_fitness = None, 0

    for _ in range(generations):
        fitness_scores = [(s, fitness_function(s, ratings_data)) for s in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        best_candidate = fitness_scores[0]

        if best_candidate[1] > best_fitness:
            best_schedule, best_fitness = best_candidate

        new_pop = [x[0] for x in fitness_scores[:elitism_size]]

        while len(new_pop) < population_size:
            p1 = random.choice(fitness_scores[:population_size // 2])[0]
            p2 = random.choice(fitness_scores[:population_size // 2])[0]

            if random.random() < crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]

            if random.random() < mutation_rate:
                c1 = mutate(c1, all_programs)
            if random.random() < mutation_rate:
                c2 = mutate(c2, all_programs)

            new_pop.extend([c1, c2])

        population = new_pop[:population_size]

    return best_schedule, best_fitness


# ============================================
# 📺 STREAMLIT FRONT-END
# ============================================

st.title("📺 Genetic Algorithm — TV Program Scheduling Optimizer")

st.info("""
🧾 **Instructions**
1. Place your file named **`program_ratings (2).csv`** in the same folder.
2. Each row = Program name + 18 rating values (6:00–23:00).
3. Press **Run All 3 Trials** to view results.
""")

file_path = 'program_ratings (2).csv'
if not os.path.exists(file_path):
    st.error("❌ File not found! Please upload or check name.")
else:
    ratings = read_csv_to_dict(file_path)
    df = pd.read_csv(file_path)
    st.subheader("📊 Program Ratings Dataset")
    st.dataframe(df)

    if ratings:
        all_programs = list(ratings.keys())
        all_time_slots = list(range(6, 24))
        SCHEDULE_LENGTH = len(all_time_slots)

        # Distinct rates for better variation
        trials = [
            ("Trial 1", 0.9, 0.5, 10),
            ("Trial 2", 0.6, 0.7, 20),
            ("Trial 3", 0.8, 0.9, 30),
        ]

        if st.button("🚀 Run All 3 Trials"):
            for label, co, mu, seed in trials:
                st.header(f"🔹 {label}")
                st.write(f"**Crossover Rate:** {co} | **Mutation Rate:** {mu}")
                random.seed(seed)
                np.random.seed(seed)
                schedule, fitness = genetic_algorithm(
                    ratings, all_programs, SCHEDULE_LENGTH,
                    crossover_rate=co, mutation_rate=mu
                )

                df_result = pd.DataFrame({
                    "Time Slot": [f"{t:02d}:00" for t in all_time_slots],
                    "Program": schedule
                })

                st.dataframe(df_result)
                st.success(f"✅ Best Fitness Score: {fitness:.2f}")
                st.markdown("---")
    else:
        st.error("⚠️ Could not load program ratings properly.")
