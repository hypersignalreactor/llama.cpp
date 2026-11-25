#pragma once

// Apollos fork · adaptive loop depth for Ouro (looped LM).
//
// With OURO_EXIT_P=<p> (0 < p < 1) a single generated token is decoded in stages: first OURO_MIN_LOOPS loops (default 1),
// then OURO_STEP loops at a time (default 1). After each stage the LM head is read, and the token stops looping once its
// top prediction's probability reaches p. A token that stops early must still have a K/V entry at every later depth for
// later tokens to attend to; OURO_FILL chooses how those entries are made:
//   copy   the last computed loop's K/V of the same layer is copied forward (free)
//   state  K/V is projected from the stage's final hidden state by each layer (about a fifth of a loop per stage)
// A later loop that does run overwrites its own slot. Multi-token batches (the prompt) run all loops in one graph.

#include <cstdint>
#include <vector>

struct ouro_stage_state {
    int   lo        = -1;    // -1: all loops in one graph; >= 0: build only loops [lo, hi)
    int   hi        = -1;
    int   step      = 1;
    int   fill      = 0;     // 0: copy, 1: state
    int   n_loops   = 1;
    float exit_p    = 0.0f;  // 0: adaptive depth off
    int   min_loops = 1;
    std::vector<float> carry;   // the normed hidden state handed from one loop to the next
    int64_t n_exit[16] = {};    // tokens that stopped after loop i+1
};

ouro_stage_state & ouro_stage();
