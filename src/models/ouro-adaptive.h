#pragma once

// Apollos fork · adaptive loop depth for Ouro (looped LM).
//
// With OURO_EXIT_P=<p> (0 < p < 1) a single generated token is decoded one loop at a time; after each loop the LM head is
// read, and the token stops looping once its top prediction's probability reaches p (never before OURO_MIN_LOOPS loops).
// The K/V a loop writes is also copied forward into every later loop's cache slot, so a token that stops early still has
// an entry at every depth for later tokens to attend to (a later loop that does run overwrites its own slot).
// Multi-token batches (the prompt) always run all loops in one graph, as before.

#include <cstdint>
#include <vector>

struct ouro_stage_state {
    int   stage     = -1;    // -1: all loops in one graph; >= 0: build only this loop
    int   n_loops   = 1;
    float exit_p    = 0.0f;  // 0: adaptive depth off
    int   min_loops = 1;
    std::vector<float> carry;   // the normed hidden state handed from one loop to the next
    int64_t n_exit[16] = {};    // tokens that stopped after loop i+1
};

ouro_stage_state & ouro_stage();
