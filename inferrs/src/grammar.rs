//! Structured output: JSON grammar-constrained token sampling.
//!
//! This module implements a byte-level finite-state machine (FSM) that
//! enforces valid JSON during token generation.  At each sampling step the FSM
//! is queried to determine which bytes are legally continuable from the current
//! partial output; any token whose decoded bytes would violate the grammar has
//! its logit set to −∞ before sampling.
//!
//! ## Grammar modes
//!
//! - **JSON object** (`response_format.type = "json_object"`): enforces that
//!   the output is a complete, valid JSON object `{…}`.
//! - **JSON schema** (`response_format.type = "json_schema"`): enforces the
//!   structural constraints of a provided JSON Schema (currently aliases
//!   JSON-object mode — full schema validation is a future extension).
//!
//! ## Design
//!
//! The FSM tracks the nested structure of the in-progress JSON value:
//!
//! ```text
//! Start → '{' → ObjectKey | '}' (empty object)
//! ObjectKey → '"' → StringInKey → '"' → ':' → Value → ',' → ObjectKey
//!                                                      → '}' → parent
//! Value → ObjectValue | ArrayValue | StringValue |
//!         NumberValue | TrueValue | FalseValue | NullValue
//! ```
//!
//! Each state exposes a set of valid *next bytes* (`valid_leading_bytes`).
//! The token masking step asks: "does there exist any prefix of this token's
//! UTF-8 byte sequence that could legally continue the current FSM state?"
//! If yes, the token is kept; otherwise its logit is set to −∞.
//!
//! This is a conservative approach (never rejects valid tokens) because we
//! apply a prefix check rather than a full-decode check.  The FSM advances
//! its state incrementally as each sampled token is decoded.

use std::fmt;

// ── FSM states ────────────────────────────────────────────────────────────────

/// The depth at which the FSM currently is.  Capped to avoid unbounded
/// allocation for deeply nested structures.
const MAX_DEPTH: usize = 64;

/// One frame on the nesting stack.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Frame {
    /// Inside a JSON object: either expecting a key, value, or close brace.
    Object(ObjectPhase),
    /// Inside a JSON array: either expecting a value or close bracket.
    Array(ArrayPhase),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ObjectPhase {
    /// Just opened `{`; expecting `"` (key) or `}` (empty object).
    ExpectKey,
    /// After `,` inside object; expecting `"` (key).
    ExpectKeyAfterComma,
    /// Inside a quoted key string.
    InKey { escaped: bool },
    /// After closing `"` of key; expecting `:`.
    ExpectColon,
    /// After `:` in object; expecting a value.
    ExpectValue,
    /// After a complete value; expecting `,` or `}`.
    AfterValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ArrayPhase {
    /// Just opened `[`; expecting value or `]`.
    ExpectValue,
    /// After `,` inside array; expecting value.
    ExpectValueAfterComma,
    /// After a complete value; expecting `,` or `]`.
    AfterValue,
}

/// Primary state of the top-level value being assembled.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TopState {
    /// Before any output — waiting to start a JSON object.
    Start,
    /// Nesting stack is active; the top of the stack determines allowed bytes.
    Nested,
    /// Inside a string value (not a key) — any UTF-8 bytes are allowed except
    /// unescaped `"` and `\`.  Tracks whether the previous char was `\`.
    StringValue { escaped: bool },
    /// Inside a number.
    Number { has_dot: bool, has_exp: bool },
    /// Matching the literal `true` — `remaining` counts bytes still to emit.
    LiteralTrue { remaining: u8 },
    /// Matching the literal `false`.
    LiteralFalse { remaining: u8 },
    /// Matching the literal `null`.
    LiteralNull { remaining: u8 },
    /// Entire top-level value is complete; no more tokens should be emitted.
    Done,
}

/// JSON FSM instance, one per in-flight sequence.
#[derive(Debug, Clone)]
pub struct JsonFsm {
    state: TopState,
    /// Nesting stack (objects and arrays).
    stack: Vec<Frame>,
    /// All bytes produced so far (for debugging / introspection).
    #[allow(dead_code)]
    produced: Vec<u8>,
}

impl JsonFsm {
    /// Create a new FSM expecting a top-level JSON object `{…}`.
    pub fn new() -> Self {
        Self {
            state: TopState::Start,
            stack: Vec::new(),
            produced: Vec::new(),
        }
    }

    /// Returns `true` if the FSM has consumed a complete, well-formed JSON
    /// value and no further tokens should be generated.
    pub fn is_done(&self) -> bool {
        self.state == TopState::Done
    }

    /// Advance the FSM by one decoded byte.
    ///
    /// Returns `true` if the byte was accepted, `false` if it violated the
    /// grammar (caller should treat this as an error / fallback).
    pub fn advance(&mut self, byte: u8) -> bool {
        self.produced.push(byte);
        match &self.state.clone() {
            TopState::Done => {
                // Whitespace after the complete value is tolerated.
                byte.is_ascii_whitespace()
            }

            TopState::Start => {
                // Skip leading whitespace, then expect `{`.
                if byte.is_ascii_whitespace() {
                    return true;
                }
                if byte == b'{' {
                    self.stack.push(Frame::Object(ObjectPhase::ExpectKey));
                    self.state = TopState::Nested;
                    return true;
                }
                false
            }

            TopState::Nested => self.advance_nested(byte),

            TopState::StringValue { escaped } => {
                let was_escaped = *escaped;
                if was_escaped {
                    // Any byte is a valid escaped character.
                    self.state = TopState::StringValue { escaped: false };
                    return true;
                }
                match byte {
                    b'"' => {
                        // String closed; pop back to parent.
                        self.close_value();
                        true
                    }
                    b'\\' => {
                        self.state = TopState::StringValue { escaped: true };
                        true
                    }
                    0x00..=0x1f => {
                        // Unescaped control characters are not valid in JSON strings.
                        false
                    }
                    _ => true, // all other UTF-8 bytes are fine
                }
            }

            TopState::Number { has_dot, has_exp } => {
                let has_dot = *has_dot;
                let has_exp = *has_exp;
                match byte {
                    b'0'..=b'9' => true,
                    b'.' if !has_dot && !has_exp => {
                        self.state = TopState::Number {
                            has_dot: true,
                            has_exp,
                        };
                        true
                    }
                    b'e' | b'E' if !has_exp => {
                        self.state = TopState::Number {
                            has_dot,
                            has_exp: true,
                        };
                        true
                    }
                    b'+' | b'-' if has_exp => true,
                    // Number terminators: delegate to parent state handling.
                    b',' | b'}' | b']' | b'\n' | b'\r' | b'\t' | b' ' => {
                        self.close_value();
                        self.advance_nested(byte)
                    }
                    _ => false,
                }
            }

            TopState::LiteralTrue { remaining } => {
                // "true" — remaining bytes: r, u, e (3, 2, 1)
                let rem = *remaining;
                let expected = match rem {
                    3 => b'r',
                    2 => b'u',
                    1 => b'e',
                    _ => return false,
                };
                if byte != expected {
                    return false;
                }
                if rem == 1 {
                    self.close_value();
                } else {
                    self.state = TopState::LiteralTrue { remaining: rem - 1 };
                }
                true
            }

            TopState::LiteralFalse { remaining } => {
                // "false" — remaining: a, l, s, e (4, 3, 2, 1)
                let rem = *remaining;
                let expected = match rem {
                    4 => b'a',
                    3 => b'l',
                    2 => b's',
                    1 => b'e',
                    _ => return false,
                };
                if byte != expected {
                    return false;
                }
                if rem == 1 {
                    self.close_value();
                } else {
                    self.state = TopState::LiteralFalse { remaining: rem - 1 };
                }
                true
            }

            TopState::LiteralNull { remaining } => {
                // "null" — remaining: u, l, l (3, 2, 1)
                let rem = *remaining;
                let expected = match rem {
                    3 => b'u',
                    2 => b'l',
                    1 => b'l',
                    _ => return false,
                };
                if byte != expected {
                    return false;
                }
                if rem == 1 {
                    self.close_value();
                } else {
                    self.state = TopState::LiteralNull { remaining: rem - 1 };
                }
                true
            }
        }
    }

    /// Advance a byte when the top state is `Nested`.
    fn advance_nested(&mut self, byte: u8) -> bool {
        let frame = match self.stack.last_mut() {
            Some(f) => f,
            None => {
                // Stack exhausted after completing the root value.
                self.state = TopState::Done;
                return byte.is_ascii_whitespace();
            }
        };

        match frame {
            Frame::Object(phase) => match phase {
                ObjectPhase::ExpectKey | ObjectPhase::ExpectKeyAfterComma => {
                    if byte.is_ascii_whitespace() {
                        return true;
                    }
                    if byte == b'"' {
                        *phase = ObjectPhase::InKey { escaped: false };
                        return true;
                    }
                    if byte == b'}' && matches!(phase, ObjectPhase::ExpectKey) {
                        self.stack.pop();
                        self.close_value();
                        return true;
                    }
                    false
                }
                ObjectPhase::InKey { escaped } => {
                    let was_escaped = *escaped;
                    if was_escaped {
                        *escaped = false;
                        return true;
                    }
                    match byte {
                        b'"' => {
                            *phase = ObjectPhase::ExpectColon;
                            true
                        }
                        b'\\' => {
                            *escaped = true;
                            true
                        }
                        0x00..=0x1f => false,
                        _ => true,
                    }
                }
                ObjectPhase::ExpectColon => {
                    if byte.is_ascii_whitespace() {
                        return true;
                    }
                    if byte == b':' {
                        *phase = ObjectPhase::ExpectValue;
                        return true;
                    }
                    false
                }
                ObjectPhase::ExpectValue => {
                    if byte.is_ascii_whitespace() {
                        return true;
                    }
                    *phase = ObjectPhase::AfterValue;
                    self.begin_value(byte)
                }
                ObjectPhase::AfterValue => {
                    if byte.is_ascii_whitespace() {
                        return true;
                    }
                    match byte {
                        b',' => {
                            *phase = ObjectPhase::ExpectKeyAfterComma;
                            true
                        }
                        b'}' => {
                            self.stack.pop();
                            self.close_value();
                            true
                        }
                        _ => false,
                    }
                }
            },

            Frame::Array(phase) => match phase {
                ArrayPhase::ExpectValue | ArrayPhase::ExpectValueAfterComma => {
                    if byte.is_ascii_whitespace() {
                        return true;
                    }
                    if byte == b']' && matches!(phase, ArrayPhase::ExpectValue) {
                        self.stack.pop();
                        self.close_value();
                        return true;
                    }
                    *phase = ArrayPhase::AfterValue;
                    self.begin_value(byte)
                }
                ArrayPhase::AfterValue => {
                    if byte.is_ascii_whitespace() {
                        return true;
                    }
                    match byte {
                        b',' => {
                            *phase = ArrayPhase::ExpectValueAfterComma;
                            true
                        }
                        b']' => {
                            self.stack.pop();
                            self.close_value();
                            true
                        }
                        _ => false,
                    }
                }
            },
        }
    }

    /// Begin a new JSON value starting with `first_byte`.  Returns `true`
    /// when the byte is a legal value-start character.
    fn begin_value(&mut self, first_byte: u8) -> bool {
        match first_byte {
            b'"' => {
                self.state = TopState::StringValue { escaped: false };
                true
            }
            b'{' => {
                if self.stack.len() >= MAX_DEPTH {
                    return false;
                }
                self.stack.push(Frame::Object(ObjectPhase::ExpectKey));
                self.state = TopState::Nested;
                true
            }
            b'[' => {
                if self.stack.len() >= MAX_DEPTH {
                    return false;
                }
                self.stack.push(Frame::Array(ArrayPhase::ExpectValue));
                self.state = TopState::Nested;
                true
            }
            b'-' | b'0'..=b'9' => {
                self.state = TopState::Number {
                    has_dot: false,
                    has_exp: false,
                };
                true
            }
            b't' => {
                self.state = TopState::LiteralTrue { remaining: 3 };
                true
            }
            b'f' => {
                self.state = TopState::LiteralFalse { remaining: 4 };
                true
            }
            b'n' => {
                self.state = TopState::LiteralNull { remaining: 3 };
                true
            }
            _ => false,
        }
    }

    /// Called when a value (string, number, literal, object, array) completes.
    /// Returns to the enclosing nested context.
    fn close_value(&mut self) {
        // If the stack is non-empty, we're inside an object or array.
        if !self.stack.is_empty() {
            // The parent frame advances its phase on the next `advance_nested` call
            // (we already popped for `}` / `]` in the caller when needed).
            // For strings and literals that close with their last byte we set
            // `Nested` so the *next* advance goes to the parent's AfterValue.
            self.state = TopState::Nested;
        } else {
            // Top-level value complete.
            self.state = TopState::Done;
        }
    }

    /// Return the set of bytes that are valid as the **first byte** of the next
    /// token given the current FSM state.
    ///
    /// Used by the token masker to pre-filter the vocabulary.
    #[allow(dead_code)]
    pub fn valid_leading_bytes(&self) -> Vec<u8> {
        let mut v: Vec<u8> = Vec::with_capacity(128);
        match &self.state {
            TopState::Done => {
                // Only trailing whitespace.
                v.extend_from_slice(b" \t\n\r");
            }
            TopState::Start => {
                v.extend_from_slice(b" \t\n\r{");
            }
            TopState::StringValue { escaped } => {
                if *escaped {
                    // After `\`: any byte is valid.
                    v.extend(0u8..=255u8);
                } else {
                    // Any printable non-control byte, plus `\`.
                    for b in 0x20u8..=0xfeu8 {
                        v.push(b);
                    }
                    v.push(0xff);
                }
            }
            TopState::Number { has_dot, has_exp } => {
                v.extend_from_slice(b"0123456789");
                if !has_dot && !has_exp {
                    v.push(b'.');
                }
                if !has_exp {
                    v.push(b'e');
                    v.push(b'E');
                }
                if *has_exp {
                    v.push(b'+');
                    v.push(b'-');
                }
                // Terminators
                v.extend_from_slice(b" \t\n\r,}]");
            }
            TopState::LiteralTrue { remaining } => {
                let b = match remaining {
                    3 => b'r',
                    2 => b'u',
                    _ => b'e',
                };
                v.push(b);
            }
            TopState::LiteralFalse { remaining } => {
                let b = match remaining {
                    4 => b'a',
                    3 => b'l',
                    2 => b's',
                    _ => b'e',
                };
                v.push(b);
            }
            TopState::LiteralNull { remaining } => {
                let b = match remaining {
                    3 => b'u',
                    2 => b'l',
                    _ => b'l',
                };
                v.push(b);
            }
            TopState::Nested => {
                self.nested_valid_leading_bytes(&mut v);
            }
        }
        v
    }

    fn nested_valid_leading_bytes(&self, v: &mut Vec<u8>) {
        let frame = match self.stack.last() {
            Some(f) => f,
            None => return,
        };
        match frame {
            Frame::Object(phase) => match phase {
                ObjectPhase::ExpectKey => {
                    v.extend_from_slice(b" \t\n\r\"");
                    v.push(b'}'); // empty object
                }
                ObjectPhase::ExpectKeyAfterComma => {
                    v.extend_from_slice(b" \t\n\r\"");
                }
                ObjectPhase::InKey { escaped } => {
                    if *escaped {
                        v.extend(0u8..=255u8);
                    } else {
                        for b in 0x20u8..=0xfeu8 {
                            v.push(b);
                        }
                        v.push(0xff);
                    }
                }
                ObjectPhase::ExpectColon => {
                    v.extend_from_slice(b" \t\n\r:");
                }
                ObjectPhase::ExpectValue => {
                    v.extend_from_slice(b" \t\n\r\"{[tfn-0123456789");
                }
                ObjectPhase::AfterValue => {
                    v.extend_from_slice(b" \t\n\r,}");
                }
            },
            Frame::Array(phase) => match phase {
                ArrayPhase::ExpectValue => {
                    v.extend_from_slice(b" \t\n\r\"{[tfn-0123456789");
                    v.push(b']'); // empty array
                }
                ArrayPhase::ExpectValueAfterComma => {
                    v.extend_from_slice(b" \t\n\r\"{[tfn-0123456789");
                }
                ArrayPhase::AfterValue => {
                    v.extend_from_slice(b" \t\n\r,]");
                }
            },
        }
    }

    /// Mask the logits vector in-place: set logit to −∞ for every token whose
    /// UTF-8 byte sequence cannot legally continue the current FSM state.
    ///
    /// Returns `true` if advancing the FSM through **all** bytes in `token`
    /// succeeds without a rejection.  The FSM state is not mutated.
    ///
    /// This is used by [`mask_logits`] to validate the complete multi-byte
    /// token before allowing it — checking only the first byte is insufficient
    /// because later bytes may violate the grammar even when the first byte
    /// passes.
    pub fn token_is_valid(&self, token: &[u8]) -> bool {
        if token.is_empty() {
            // EOS: only valid when the FSM has reached Done.
            return self.is_done();
        }
        let mut probe = self.clone();
        for &byte in token {
            if !probe.advance(byte) {
                return false;
            }
        }
        true
    }

    /// Mask `logits` in-place: set logit to −∞ for every token whose decoded
    /// byte sequence cannot legally continue the current FSM state.
    ///
    /// Every byte in a token's UTF-8 representation is validated against a
    /// cloned FSM snapshot so that multi-byte tokens whose later bytes violate
    /// the grammar are correctly rejected, not just their first byte.
    ///
    /// `token_bytes[i]` should be the decoded UTF-8 byte string for token `i`.
    /// Tokens with index ≥ `logits.len()` are ignored.
    pub fn mask_logits(&self, logits: &mut [f32], token_bytes: &[Vec<u8>]) {
        for (i, logit) in logits.iter_mut().enumerate() {
            if *logit == f32::NEG_INFINITY {
                continue; // already masked — skip the clone overhead
            }
            let bytes = match token_bytes.get(i) {
                Some(b) => b,
                None => continue,
            };
            if !self.token_is_valid(bytes) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
}

impl Default for JsonFsm {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for JsonFsm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "JsonFsm(state={:?}, depth={})",
            self.state,
            self.stack.len()
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn advance_str(fsm: &mut JsonFsm, s: &str) -> bool {
        for b in s.bytes() {
            if !fsm.advance(b) {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_empty_object() {
        let mut fsm = JsonFsm::new();
        assert!(advance_str(&mut fsm, "{}"));
        assert!(fsm.is_done());
    }

    #[test]
    fn test_simple_object() {
        let mut fsm = JsonFsm::new();
        assert!(advance_str(&mut fsm, r#"{"key": "value"}"#));
        assert!(fsm.is_done());
    }

    #[test]
    fn test_nested_object() {
        let mut fsm = JsonFsm::new();
        assert!(advance_str(&mut fsm, r#"{"a": {"b": 42}}"#));
        assert!(fsm.is_done());
    }

    #[test]
    fn test_array_value() {
        let mut fsm = JsonFsm::new();
        assert!(advance_str(&mut fsm, r#"{"items": [1, 2, 3]}"#));
        assert!(fsm.is_done());
    }

    #[test]
    fn test_boolean_and_null() {
        let mut fsm = JsonFsm::new();
        assert!(advance_str(
            &mut fsm,
            r#"{"a": true, "b": false, "c": null}"#
        ));
        assert!(fsm.is_done());
    }

    #[test]
    fn test_invalid_start() {
        let mut fsm = JsonFsm::new();
        assert!(!advance_str(&mut fsm, "invalid"));
    }

    #[test]
    fn test_partial_is_not_done() {
        let mut fsm = JsonFsm::new();
        advance_str(&mut fsm, r#"{"key": "#);
        assert!(!fsm.is_done());
    }

    #[test]
    fn test_valid_leading_bytes_start() {
        let fsm = JsonFsm::new();
        let valid = fsm.valid_leading_bytes();
        assert!(valid.contains(&b'{'));
        assert!(valid.contains(&b' '));
        assert!(!valid.contains(&b'['));
    }

    #[test]
    fn test_number_value() {
        let mut fsm = JsonFsm::new();
        assert!(advance_str(&mut fsm, r#"{"n": 3.14}"#));
        assert!(fsm.is_done());
    }

    #[test]
    fn test_string_with_escape() {
        let mut fsm = JsonFsm::new();
        assert!(advance_str(&mut fsm, r#"{"s": "hello \"world\""}"#));
        assert!(fsm.is_done());
    }
}
