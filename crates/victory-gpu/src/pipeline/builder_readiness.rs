//! Check only the builder pipelines the title creates, using the real
//! translation, root signature and D3D12 pipeline creation paths.
//! The audit takes precedence over builder execution in `builder_runs`.

use std::sync::{Mutex, OnceLock};

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_READINESS").is_some_and(|v| v == "1"))
}

#[derive(Default)]
struct Readiness {
    created: usize,
    ready: usize,
    failed: usize,
}

impl Readiness {
    fn counts(&self) -> (usize, &'static str) {
        let pending = self.created - self.ready - self.failed;
        let status = if self.failed != 0 {
            "unavailable"
        } else if pending == 0 && self.created != 0 {
            "all created PSOs ready"
        } else {
            "checking"
        };
        (pending, status)
    }

    fn report(&self, shader: &str, event: &str, reason: &str) {
        let (pending, status) = self.counts();
        tracing::info!(
            shader,
            event,
            reason,
            status,
            created = self.created,
            ready = self.ready,
            failed = self.failed,
            pending,
            frame = crate::frame::current(),
            "BVH readiness (dispatch disabled)"
        );
    }
}

static TOTALS: Mutex<Readiness> = Mutex::new(Readiness {
    created: 0,
    ready: 0,
    failed: 0,
});

pub(super) fn created(shader: &str) {
    let mut totals = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    totals.created += 1;
    totals.report(shader, "registered", "");
}

pub(super) fn checked(shader: &str, success: bool, reason: &str) {
    let mut totals = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    if success {
        totals.ready += 1;
        totals.report(shader, "ready", "");
    } else {
        totals.failed += 1;
        totals.report(
            shader,
            "failed",
            if reason.is_empty() {
                "host binding layout unavailable"
            } else {
                reason
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_readiness_reports_the_exact_shader_error_without_touching_the_device() {
        use std::sync::Arc;
        let shader = Arc::new(super::super::Shader {
            stage: "compute",
            hash: "fixture".into(),
            container: Vec::new(),
            translated: OnceLock::from(Err("instruction at 0x258".into())),
        });
        let mut reason = String::new();
        let pipeline = super::super::host::HostPipeline::new(&[shader], None, Some(&mut reason));
        assert!(pipeline.is_none());
        assert_eq!(reason, "shader translation: instruction at 0x258");
    }

    #[test]
    fn builder_readiness_overrides_execution_even_when_both_switches_are_set() {
        const CHILD: &str = "VICTORY_TEST_BVH_READINESS_CHILD";
        if std::env::var_os(CHILD).is_some() {
            assert!(enabled());
            assert!(!super::super::builder_runs());
            return;
        }
        // Test in a fresh process: both modes cache their environment, and a
        // test must not mutate it underneath other parallel tests.
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "pipeline::builder_readiness::tests::builder_readiness_overrides_execution_even_when_both_switches_are_set"])
            .env(CHILD, "1")
            .env("VICTORY_BVH_BUILDER", "1")
            .env("VICTORY_BVH_READINESS", "1")
            .status().unwrap();
        assert!(result.success());
    }

    #[test]
    fn builder_readiness_changes_with_later_created_stages_and_keeps_failures() {
        let mut r = Readiness::default();
        assert_eq!(r.counts(), (0, "checking"));
        r.created = 2;
        r.ready = 1;
        assert_eq!(r.counts(), (1, "checking"));
        r.ready += 1;
        assert_eq!(r.counts(), (0, "all created PSOs ready"));
        r.created += 1;
        assert_eq!(r.counts(), (1, "checking"));
        r.failed += 1;
        assert_eq!(r.counts(), (0, "unavailable"));
        r.created += 1;
        assert_eq!(r.counts(), (1, "unavailable"));
    }
}
