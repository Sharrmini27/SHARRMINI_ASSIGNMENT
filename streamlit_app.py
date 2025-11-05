import streamlit as st
import pandas as pd
import random
import csv
import numpy as np
import os

# --- GENETIC ALGORITHM ENGINE ---

def read_csv_to_dict(file_path):
    """Reads CSV and returns a dictionary with program names and their ratings."""
    program_ratings = {}
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader, None)  # Skip header if present

            for row in reader:
                if len(row) > 1:
                    program = row[0]
                    try:
                        ratings = [float(x) for x in row[1:]]
                        program_ratings[program] = ratings
                    except ValueError:
                        st.warning(f"⚠️ Skipping row for '{program}' due to invalid data.")
        return program_ratings

    except FileNotFoundError:
        st.error(f"❌ File '{file_path}' not found! Please upload it to the same directory.")
        return {}

def fitness_function(schedule, ratings_data):
    """Calculates the total fitness for a schedule based on ratings."""
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        if program in ratings_data and time_slot < len(ratings_data[program]):
            total_rating += ratings_data[program][time_slot]
    return total_rating

def create_random_schedule(all_programs, schedule_length):
    return [random.choice(all_programs) for _ in range(schedule_length)]

def crossover(schedule1, schedule2, schedule_length):
    """Performs single-point crossover."""
    if len(schedule1) < 2 or len(schedule2) < 2:
        return schedule1, schedule2
    crossover_point = random.randint(1, schedule_length - 1)
    child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
    child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
    return child1, child2

def mutate(schedule, all_programs, schedule_length):
    """Performs mutation by replacing one random slot with a new random program."""
    schedule_copy = schedule.copy()
    mutation_point = random.randint(0, schedule_length - 1)
    new_program = random.choice(all_programs)
    schedule_copy[mutation_point] = new_program
    return schedule_copy

def genetic_algorithm(ratings_data, all_programs, schedule_length,
                      generations=100, population_size=50,
                      crossover_rate=0.9, mutation_rate=0.1, elitism_size=2):
    """Main GA routine."""
    population = [create_random_schedule(all_programs, schedule_length) for _ in range(population_size)]
    best_schedule = None
    best_fitness = 0

    for generation in range(generations):
        pop_with_fitness = [(schedule, fitness_function(schedule, ratings_data)) for schedule in population]
        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)

        if pop_with_fitness[0][1] > best_fitness:
            best_schedule, best_fitness = pop_with_fitness[0]

        new_population = [x[0] for x in pop_with_fitness[:elitism_size]]  # elitism

        while len(new_population) < population_size:
            parent1 = random.choice(pop_with_fitness[:population_size // 2])[0]
            parent2 = random.choice(pop_with_fitness[:population_size // 2])[0]

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2, schedule_length)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1, all_programs, schedule_length)
            if random.random() < mutation_rate:
                child2 = mutate(child2, all_programs, schedule_length)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    return best_schedule, best_fitness

# --- STREAMLIT INTERFACE ---

st.title("📺 TV Program Scheduling Optimization using Genetic Algorithm")
st.write("This app optimizes TV program schedules based on audience ratings using a Genetic Algorithm.")

# CSV file handling
file_path = 'program_ratings (2).csv'
if not os.path.exists(file_path):
    st.warning("⚠️ CSV file not found! Please upload 'program_ratings (2).csv' below.")
    uploaded_file = st.file_uploader("Upload your program_ratings (2).csv", type=["csv"])
    if uploaded_file:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ File uploaded successfully! Please rerun the app.")
        st.stop()

ratings = read_csv_to_dict(file_path)

if not ratings:
    st.stop()

df_display = pd.read_csv(file_path)
st.subheader("📊 Program Ratings Dataset")
st.dataframe(df_display)

all_programs = list(ratings.keys())
time_slots = list(range(6, 24))
SCHEDULE_LENGTH = len(time_slots)

# Sidebar parameters
st.sidebar.header("🧬 Genetic Algorithm Parameters")

# Updated mutation and crossover rate ranges
crossover_rate = st.sidebar.slider("Crossover Rate", 0.6, 0.95, 0.85, 0.05)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.15, 0.05, 0.01)
generations = st.sidebar.slider("Number of Generations", 50, 300, 150, 10)
population_size = st.sidebar.slider("Population Size", 20, 100, 50, 10)

if st.button("🚀 Run Genetic Algorithm"):
    best_schedule, best_fitness = genetic_algorithm(
        ratings_data=ratings,
        all_programs=all_programs,
        schedule_length=SCHEDULE_LENGTH,
        generations=generations,
        population_size=population_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate
    )

    st.success("✅ Optimization Complete!")
    df_result = pd.DataFrame({
        "Time Slot": [f"{h:02d}:00" for h in time_slots],
        "Program": best_schedule
    })
    st.dataframe(df_result)
    st.write(f"**Best Fitness Score:** {best_fitness:.2f}")
