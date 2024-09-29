# Quantum Inspired ✨

Welcome to the **Quantum Inspired** section, where we dive deep into the magical world of **tensor networks** and other quantum mechanical principles in search of machine learning breakthroughs. Here, you'll find a collection of techniques that *sound* like they should outperform classical methods, but (spoiler alert!) often don’t... at least not yet. But hey, if we're having fun along the way and getting better than 80% accuracy, it’s a win, right? 😄

## Overview

In this folder, I experiment with **quantum-inspired approaches** to machine learning. These techniques borrow from the elegant math of quantum mechanics—like tensor networks, state optimization, and entanglement—and apply them to practical machine learning problems. 

So far, I’ve implemented:
- **MPS Classifier on MNIST**: A full-fledged MPS (Matrix Product States) model with gradient optimization.
- **Simplified State Averaging Classifier**: A more down-to-earth version where I simply compute the average quantum state for each digit class, achieving surprisingly solid results with **80%+ accuracy**. Who knew averaging could be so effective?

### What's Inside?

This folder will soon house multiple quantum-inspired techniques, but for now, you can find:

1. **MPS Classifier (Gradient-Optimized)**:
   - This method uses a full tensor network with gradient descent to classify digits from the MNIST dataset. Think of it like going for a long, scenic road trip through the universe of quantum states.
   - It’s sophisticated, precise, and... well, it usually performs about as well as more classical approaches—but hey, it’s quantum, so that’s cool, right?

2. **Simplified Averaged Quantum State Classifier**:
   - Instead of optimizing individual states for each training example, this approach averages the quantum states of all training samples for each digit class. The result is a clean, simplified quantum state for each digit. 
   - The kicker? This less “quantum fancy” version performs surprisingly well, with **80%+ accuracy** on MNIST. Turns out, sometimes less is more!

---

## Current Results

| **Method**                         | **Accuracy**      |
|------------------------------------ |------------------ |
| MPS Classifier (Gradient Optimized) | ~94%              |
| Averaged Quantum State Classifier   | ~80%              |

It’s worth noting that while these quantum-inspired methods tend to perform on par (or slightly below) classical models, they offer unique ways of thinking about data—ways that could become more powerful as quantum hardware and software evolve. For now, we enjoy the intellectual exercise and the journey into the quantum realm. 🚀

---

## What’s Next?

Expect to see more approaches soon! Here are a few ideas I'm working on:

- **PEPS (Projected Entangled Pair States)**: Extending the MPS framework into two dimensions for even more complex models.
- **Quantum Kernel Methods**: Exploring quantum kernel techniques for classification.
- **Variational Quantum Circuits**: Trying out hybrid quantum-classical models that optimize quantum circuits for better ML performance.

---

## Why Quantum (Even If It Works Just as Well)?

Because it’s **quantum**, and sometimes you just want to feel like a wizard casting spells on your data. ⚡ Besides, as the world shifts towards quantum computing, these methods could become crucial for leveraging the full potential of quantum algorithms in machine learning.

---

## Stay Tuned!

This folder will be regularly updated with new approaches, both simple and complex. Whether they work better than classical methods or not, at least they’re fun to try! 

Got any suggestions or want to collaborate? Feel free to reach out.
