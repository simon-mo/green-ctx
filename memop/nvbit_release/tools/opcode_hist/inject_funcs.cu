/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"


extern "C" __device__ __noinline__ void count_instrs(
                                                    //  int predicate,
                                                    //  int instr_type,
                                                    //  int count_warp_level,
                                                    //  uint64_t p_hist,
                                                     uint32_t sleep_ns
                                                     ) {
    __nanosleep(sleep_ns);

    // /* all the active threads will compute the active mask */
    // const int active_mask = __ballot_sync(__activemask(), 1);
    // /* compute the predicate mask */
    // const int predicate_mask = __ballot_sync(__activemask(), predicate);
    // /* each thread will get a lane id (get_lane_id is in utils/utils.h) */
    // const int laneid = get_laneid();
    // /* get the id of the first active thread */
    // const int first_laneid = __ffs(active_mask) - 1;
    // /* count all the active thread */
    // const int num_threads = __popc(predicate_mask);
    // /* only the first active thread will perform the atomic */
    // if (first_laneid == laneid) {
    //     uint64_t* hist = (uint64_t*)p_hist;
    //     if (count_warp_level) {
    //         /* num threads can be zero when accounting for predicates off */
    //         if (num_threads > 0)
    //             atomicAdd((unsigned long long*)&hist[instr_type], 1);
    //     } else {
    //         atomicAdd((unsigned long long*)&hist[instr_type], num_threads);
    //     }
    // }
}
