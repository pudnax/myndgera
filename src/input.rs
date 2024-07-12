use glam::{vec2, Vec2};
use winit::{
    dpi::PhysicalPosition,
    event::{DeviceEvent, ElementState, KeyEvent, MouseScrollDelta, RawKeyEvent},
    keyboard::{KeyCode, PhysicalKey},
};

#[derive(Clone, Copy, Debug, Default)]
pub struct MouseState {
    pub screen_position: Vec2,
    pub delta: Vec2,
    pub scroll: f32,
    pub buttons_held: u32,
    pub buttons_pressed: u32,
    pub buttons_released: u32,
}

impl MouseState {
    pub const LEFT: u32 = 0;
    pub const MIDDLE: u32 = 1;
    pub const RIGHT: u32 = 2;

    pub fn refresh(&mut self) {
        self.delta = vec2(0., 0.);
        self.scroll = 0.;
        self.buttons_pressed = 0;
        self.buttons_released = 0;
    }

    pub fn left_pressed(&self) -> bool {
        self.buttons_pressed & (1 << Self::LEFT) != 0
    }
    pub fn middle_pressed(&self) -> bool {
        self.buttons_pressed & (1 << Self::MIDDLE) != 0
    }
    pub fn right_pressed(&self) -> bool {
        self.buttons_pressed & (1 << Self::RIGHT) != 0
    }
    pub fn left_released(&self) -> bool {
        self.buttons_released & (1 << Self::LEFT) != 0
    }
    pub fn middle_released(&self) -> bool {
        self.buttons_released & (1 << Self::MIDDLE) != 0
    }
    pub fn right_released(&self) -> bool {
        self.buttons_released & (1 << Self::RIGHT) != 0
    }
    pub fn left_held(&self) -> bool {
        self.buttons_held & (1 << Self::LEFT) != 0
    }
    pub fn middle_held(&self) -> bool {
        self.buttons_held & (1 << Self::MIDDLE) != 0
    }
    pub fn right_held(&self) -> bool {
        self.buttons_held & (1 << Self::RIGHT) != 0
    }
}

#[derive(Debug, Default)]
pub struct Input {
    pub mouse_state: MouseState,
    pub move_forward: f32,
    pub move_backward: f32,
    pub move_right: f32,
    pub move_left: f32,
    pub move_up: f32,
    pub move_down: f32,
    pub boost: f32,
}

impl Input {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn update_window_input(&mut self, key_event: &KeyEvent) {
        let pressed = (key_event.state == ElementState::Pressed) as u32 as f32;
        if let PhysicalKey::Code(key) = key_event.physical_key {
            match key {
                KeyCode::KeyA => self.move_left = pressed,
                KeyCode::KeyD => self.move_right = pressed,
                KeyCode::KeyS => self.move_backward = pressed,
                KeyCode::KeyW => self.move_forward = pressed,
                KeyCode::Period | KeyCode::KeyQ => self.move_down = pressed,
                KeyCode::Slash | KeyCode::KeyE => self.move_up = pressed,
                _ => {}
            }
        }
    }

    pub fn update_device_input(&mut self, event: DeviceEvent) {
        match event {
            DeviceEvent::MouseWheel { delta, .. } => {
                self.mouse_state.scroll = -match delta {
                    MouseScrollDelta::LineDelta(_, scroll) => scroll,
                    MouseScrollDelta::PixelDelta(PhysicalPosition { y: scroll, .. }) => {
                        scroll as f32
                    }
                };
            }
            DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                self.mouse_state.delta = vec2(dx as _, dy as _);
            }
            DeviceEvent::Button { button, state } => {
                let button_id = {
                    let button = match button {
                        3 => MouseState::RIGHT,
                        2 => MouseState::MIDDLE,
                        1 => MouseState::LEFT,
                        _ => return,
                    };
                    1 << button
                };
                if let ElementState::Pressed = state {
                    self.mouse_state.buttons_held |= button_id;
                    self.mouse_state.buttons_pressed |= button_id;
                } else {
                    self.mouse_state.buttons_held &= !button_id;
                    self.mouse_state.buttons_released |= button_id;
                }
            }
            DeviceEvent::Key(RawKeyEvent {
                physical_key: PhysicalKey::Code(key),
                state,
            }) => {
                let pressed = (state == ElementState::Pressed) as u32 as f32;
                match key {
                    KeyCode::ArrowLeft => self.move_left = pressed,
                    KeyCode::ArrowRight => self.move_right = pressed,
                    KeyCode::ArrowDown => self.move_backward = pressed,
                    KeyCode::ArrowUp => self.move_forward = pressed,
                    KeyCode::ShiftLeft | KeyCode::ShiftRight => self.boost = pressed,
                    _ => {}
                }
            }
            _ => {}
        }
    }

    pub fn process_position(&self, pos: &mut [f32; 3]) {
        let dx = 0.01 * 4.0f32.powf(self.boost);
        let move_right = self.move_right - self.move_left;
        let move_up = self.move_up - self.move_down;
        let move_fwd = self.move_forward - self.move_backward;
        pos[0] += dx * move_right;
        pos[1] += dx * move_fwd;
        pos[2] += dx * move_up;
    }
}
