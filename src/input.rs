use ahash::AHashMap;
use glam::{vec2, Vec2};
use winit::{
    event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent},
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

pub type Action = &'static str;

pub struct KeyAction {
    action: Action,
    multiplier: f32,
}

impl KeyAction {
    pub fn new(action: Action, multiplier: f32) -> Self {
        Self { action, multiplier }
    }
}

pub struct KeyboardMap {
    bindings: Vec<(KeyCode, KeyAction)>,
}

impl Default for KeyboardMap {
    fn default() -> Self {
        Self::new()
    }
}

impl From<(Action, f32)> for KeyAction {
    fn from((action, multiplier): (Action, f32)) -> Self {
        Self { action, multiplier }
    }
}

impl KeyboardMap {
    pub fn new() -> Self {
        Self {
            bindings: Default::default(),
        }
    }

    pub fn bind(mut self, key: KeyCode, map: impl Into<KeyAction>) -> Self {
        self.bindings.push((key, map.into()));
        self
    }

    pub fn map(&mut self, keyboard: &KeyboardState) -> AHashMap<Action, f32> {
        let mut result: AHashMap<Action, f32> = AHashMap::new();

        for (key, s) in &mut self.bindings {
            let activation = if keyboard.is_down(*key) { 1.0 } else { 0.0 };
            *result.entry(s.action).or_default() += activation * s.multiplier;
        }

        for value in result.values_mut() {
            *value = value.clamp(-1.0, 1.0);
        }

        result
    }
}

#[derive(Clone, Debug, Default)]
pub struct KeyState {
    pub ticks: u32,
}

#[derive(Default, Clone, Debug)]
pub struct KeyboardState {
    keys_down: AHashMap<KeyCode, KeyState>,
}

impl KeyboardState {
    pub fn is_down(&self, key: KeyCode) -> bool {
        self.get_down(key).is_some()
    }

    pub fn is_down_f32(&self, key: KeyCode) -> f32 {
        self.get_down(key).is_some() as u32 as f32
    }

    pub fn was_just_pressed(&self, key: KeyCode) -> bool {
        self.get_down(key).map(|s| s.ticks == 1).unwrap_or_default()
    }

    pub fn get_down(&self, key: KeyCode) -> Option<&KeyState> {
        self.keys_down.get(&key)
    }
}

#[derive(Debug, Default)]
pub struct Input {
    pub keyboard_state: KeyboardState,
    pub mouse_state: MouseState,
}

impl Input {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn tick(&mut self) {
        self.keyboard_state.keys_down.values_mut().for_each(|val| {
            val.ticks = val.ticks.wrapping_add(1);
        });
    }

    pub fn update_on_window_input(&mut self, event: &WindowEvent) {
        let mouse = &mut self.mouse_state;
        let keyb = &mut self.keyboard_state.keys_down;

        match event {
            WindowEvent::MouseInput { state, button, .. } => {
                let button_id = {
                    let button = match button {
                        MouseButton::Left => MouseState::LEFT,
                        MouseButton::Middle => MouseState::MIDDLE,
                        MouseButton::Right => MouseState::RIGHT,
                        _ => return,
                    };
                    1 << button
                };
                if let ElementState::Pressed = state {
                    mouse.buttons_held |= button_id;
                    mouse.buttons_pressed |= button_id;
                } else {
                    mouse.buttons_held &= !button_id;
                    mouse.buttons_released |= button_id;
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key_code),
                        state,
                        ..
                    },
                ..
            } => {
                if state == &ElementState::Pressed {
                    keyb.entry(*key_code).or_insert(KeyState { ticks: 0 });
                } else {
                    keyb.remove(key_code);
                }
            }
            _ => {}
        }
    }

    pub fn update_on_device_input(&mut self, event: DeviceEvent) {
        match event {
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
            _ => {}
        }
    }

    pub fn process_position(&self, pos: &mut [f32; 3]) {
        let keyb = &self.keyboard_state;
        let key = |key| keyb.is_down_f32(key);

        let boost =
            (keyb.is_down(KeyCode::ShiftLeft) | keyb.is_down(KeyCode::ShiftRight)) as u32 as f32;
        let dx = 0.01 * 4.0f32.powf(boost);
        let move_right = key(KeyCode::ArrowRight) - key(KeyCode::ArrowLeft);
        let move_up = key(KeyCode::ArrowUp) - key(KeyCode::ArrowDown);
        let move_fwd = key(KeyCode::Period) - key(KeyCode::Slash);
        pos[0] += dx * move_right;
        pos[1] += dx * move_up;
        pos[2] += dx * move_fwd;
    }
}
