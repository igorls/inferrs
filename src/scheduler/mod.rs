//! Request scheduler with continuous batching.
//!
//! Like vLLM's scheduler, this implements iteration-level continuous batching:
//! - Running requests (in decode) are scheduled first
//! - Waiting requests (new, need prefill) get remaining budget
//! - FCFS ordering for waiting queue
//! - Preemption when cache is exhausted
//!
//! All internal buffers use GrowVec for the C++ vector semantics.

use crate::buffer::GrowVec;
use crate::cache::{KvCacheManager, SequenceCache};
use crate::sampling::SamplingParams;
use std::collections::VecDeque;
use uuid::Uuid;

/// State of a request in the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    /// Waiting to be scheduled (new or preempted).
    Waiting,
    /// Currently running (in prefill or decode).
    Running,
    /// Finished generation.
    Finished,
}

/// A single inference request tracked by the scheduler.
pub struct Request {
    pub id: String,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: GrowVec<u32>,
    pub sampling_params: SamplingParams,
    pub state: RequestState,
    pub cache: SequenceCache,
    /// How many prompt tokens have been processed.
    pub num_computed_prompt_tokens: usize,
    /// Whether generation is complete.
    pub is_finished: bool,
    /// The finish reason if finished.
    pub finish_reason: Option<String>,
    /// One-shot channel to notify the server when a new token is produced.
    pub token_sender: Option<tokio::sync::mpsc::UnboundedSender<GenerationOutput>>,
}

/// Output from the scheduler for one step.
pub struct SchedulerOutput {
    /// Requests to process this step, with their token counts.
    pub scheduled: GrowVec<ScheduledRequest>,
    /// Requests that just finished.
    pub finished_ids: GrowVec<String>,
}

/// A request scheduled for this step.
pub struct ScheduledRequest {
    /// Index into the scheduler's request storage.
    pub request_idx: usize,
    /// Number of new tokens to process this step.
    pub num_new_tokens: usize,
    /// Whether this is a prefill (true) or decode (false) step.
    pub is_prefill: bool,
}

/// Token output from generation.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    #[allow(dead_code)]
    pub token_id: u32,
    pub text: String,
    pub is_finished: bool,
    pub finish_reason: Option<String>,
}

/// The scheduler owns all requests and decides what to run each step.
pub struct Scheduler {
    /// All active requests.
    requests: GrowVec<Request>,
    /// Indices of requests in Waiting state (FCFS order).
    waiting_queue: VecDeque<usize>,
    /// Indices of requests in Running state.
    running_indices: GrowVec<usize>,
    /// Config
    max_batch_size: usize,
    max_tokens_per_step: usize,
    /// Reusable buffers for scheduler output (never shrink).
    scheduled_buf: GrowVec<ScheduledRequest>,
    finished_buf: GrowVec<String>,
}

impl Scheduler {
    pub fn new(max_batch_size: usize, max_tokens_per_step: usize) -> Self {
        Self {
            requests: GrowVec::new(),
            waiting_queue: VecDeque::new(),
            running_indices: GrowVec::new(),
            max_batch_size,
            max_tokens_per_step,
            scheduled_buf: GrowVec::new(),
            finished_buf: GrowVec::new(),
        }
    }

    /// Add a new request.
    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        token_sender: tokio::sync::mpsc::UnboundedSender<GenerationOutput>,
    ) -> String {
        let id = Uuid::new_v4().to_string();
        let request = Request {
            id: id.clone(),
            prompt_tokens,
            generated_tokens: GrowVec::new(),
            sampling_params,
            state: RequestState::Waiting,
            cache: SequenceCache::new(),
            num_computed_prompt_tokens: 0,
            is_finished: false,
            finish_reason: None,
            token_sender: Some(token_sender),
        };
        let idx = self.requests.len();
        self.requests.push(request);
        self.waiting_queue.push_back(idx);
        tracing::debug!("added request {} (idx={})", id, idx);
        id
    }

    /// Run one scheduling step. Returns which requests to execute.
    pub fn schedule(&mut self, cache_manager: &mut KvCacheManager) -> SchedulerOutput {
        self.scheduled_buf.clear();
        self.finished_buf.clear();

        let mut token_budget = self.max_tokens_per_step;
        let mut batch_size = 0;

        // 1. Schedule running requests first (decode step = 1 token each)
        let mut i = 0;
        while i < self.running_indices.len() {
            let req_idx = self.running_indices[i];
            let request = &self.requests[req_idx];

            if request.is_finished {
                // Remove from running
                self.running_indices.swap_remove(i);
                self.finished_buf.push(request.id.clone());
                continue;
            }

            if batch_size >= self.max_batch_size || token_budget == 0 {
                break;
            }

            // Decode step: 1 new token
            let num_new = 1;
            // Try to allocate cache for the new token
            let request = &mut self.requests[req_idx];
            if cache_manager
                .allocate_slots(&mut request.cache, num_new)
                .is_ok()
            {
                self.scheduled_buf.push(ScheduledRequest {
                    request_idx: req_idx,
                    num_new_tokens: num_new,
                    is_prefill: false,
                });
                token_budget = token_budget.saturating_sub(num_new);
                batch_size += 1;
                i += 1;
            } else {
                // Preempt: move to waiting queue
                tracing::warn!("preempting request {} due to cache pressure", request.id);
                request.state = RequestState::Waiting;
                cache_manager.free_sequence(&mut request.cache);
                request.num_computed_prompt_tokens = 0;
                self.waiting_queue.push_back(req_idx);
                self.running_indices.swap_remove(i);
            }
        }

        // 2. Schedule waiting requests (prefill)
        while let Some(&req_idx) = self.waiting_queue.front() {
            if batch_size >= self.max_batch_size || token_budget == 0 {
                break;
            }

            let request = &self.requests[req_idx];
            let remaining_prompt = request.prompt_tokens.len() - request.num_computed_prompt_tokens;
            let num_new = remaining_prompt.min(token_budget);

            if num_new == 0 {
                break;
            }

            // Try to allocate cache
            let request = &mut self.requests[req_idx];
            if cache_manager
                .allocate_slots(&mut request.cache, num_new)
                .is_ok()
            {
                self.waiting_queue.pop_front();
                request.state = RequestState::Running;
                request.num_computed_prompt_tokens += num_new;

                let is_full_prefill =
                    request.num_computed_prompt_tokens == request.prompt_tokens.len();
                self.scheduled_buf.push(ScheduledRequest {
                    request_idx: req_idx,
                    num_new_tokens: num_new,
                    is_prefill: true,
                });

                if is_full_prefill {
                    self.running_indices.push(req_idx);
                } else {
                    // Chunked prefill: still waiting for more prompt tokens
                    self.waiting_queue.push_front(req_idx);
                }

                token_budget = token_budget.saturating_sub(num_new);
                batch_size += 1;
            } else {
                // Can't allocate for this request, stop scheduling waiting
                break;
            }
        }

        // Swap buffers out so the caller owns the output. The empty GrowVec
        // we swap in retains zero capacity, but when schedule() is called
        // next, it will swap *back* the now-cleared buffers from the
        // previous SchedulerOutput (see take_back_buffers). For the very
        // first call, the temporary GrowVec::new() is used once and then
        // replaced with the warmed-up buffer on the next cycle.
        let scheduled = std::mem::take(&mut self.scheduled_buf);
        let finished_ids = std::mem::take(&mut self.finished_buf);
        SchedulerOutput {
            scheduled,
            finished_ids,
        }
    }

    /// Return the output buffers so they can be reused next step.
    /// This preserves the allocated capacity (never-shrink semantics).
    pub fn return_buffers(&mut self, mut output: SchedulerOutput) {
        // Clear and swap back -- capacity is preserved.
        output.scheduled.clear();
        output.finished_ids.clear();
        self.scheduled_buf = output.scheduled;
        self.finished_buf = output.finished_ids;
    }

    /// Get a reference to a request by index.
    pub fn get_request(&self, idx: usize) -> &Request {
        &self.requests[idx]
    }

    /// Get a mutable reference to a request by index.
    pub fn get_request_mut(&mut self, idx: usize) -> &mut Request {
        &mut self.requests[idx]
    }

    /// Mark a request as finished.
    pub fn finish_request(&mut self, idx: usize, reason: &str) {
        let request = &mut self.requests[idx];
        request.is_finished = true;
        request.state = RequestState::Finished;
        request.finish_reason = Some(reason.to_string());
    }

    /// Check if there are any active (waiting or running) requests.
    pub fn has_active_requests(&self) -> bool {
        !self.waiting_queue.is_empty() || !self.running_indices.is_empty()
    }

    /// Number of waiting requests.
    #[allow(dead_code)]
    pub fn num_waiting(&self) -> usize {
        self.waiting_queue.len()
    }

    /// Number of running requests.
    #[allow(dead_code)]
    pub fn num_running(&self) -> usize {
        self.running_indices.len()
    }
}
