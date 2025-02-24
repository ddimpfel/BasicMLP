#include "SimpleWindow.hpp"
#include <string>
#include <SFML/Graphics/Color.hpp>
#include <SFML/Graphics/Drawable.hpp>
#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/View.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Window/Event.hpp>
#include <SFML/Window/Keyboard.hpp>
#include <SFML/Window/VideoMode.hpp>
#include <SFML/Window/WindowEnums.hpp>
#include <imgui-SFML.h>

SimpleWindow::SimpleWindow() { setup("Window", sf::Vector2u(640, 480)); }
SimpleWindow::SimpleWindow(const std::string& l_title, const sf::Vector2u& l_size)
{
	setup(l_title, l_size);
}
SimpleWindow::~SimpleWindow() { destroy(); }

void SimpleWindow::setup(const std::string& l_title, const sf::Vector2u& l_size)
{
	m_windowTitle = l_title;
	m_uWindowSize = l_size;
	m_fWindowSize = static_cast<sf::Vector2f>(l_size);
	m_isFullscreen = false;
	m_isOpen = true;
	create();
};

void SimpleWindow::create()
{
	auto state = (m_isFullscreen ? sf::State::Fullscreen : sf::State::Windowed);
	m_window.create(sf::VideoMode({ m_uWindowSize.x, m_uWindowSize.y }),
		m_windowTitle, state, {0, 0, 8});
}

void SimpleWindow::destroy()
{
	m_window.close();
}

void SimpleWindow::processEvents(sf::View& mainView)
{
	while (const auto event = m_window.pollEvent()) 
	{
		ImGui::SFML::ProcessEvent(m_window, *event);

		if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) 
		{
			if (keyPressed->scancode == sf::Keyboard::Scancode::F5)
				toggleFullscreen();
		}
		else if (const auto* resize = event->getIf<sf::Event::Resized>()) 
		{
			// The view's size is independent of the window, so this will "zoom" accordingly.
			// If the window is enlarged, it will "zoom" out, keeping the relative size the same.
			mainView.setSize(sf::Vector2f(resize->size));
		}
		else if (event->is<sf::Event::Closed>())
		{
			m_isOpen = false;
		}
	}

	m_window.setView(mainView);
}

void SimpleWindow::toggleFullscreen()
{
	m_isFullscreen = !m_isFullscreen;
	destroy();
	create();
}

void SimpleWindow::setFramerate(unsigned int l_limit)
{
	m_framerate = l_limit;
	m_window.setFramerateLimit(l_limit);
}

void SimpleWindow::beginDraw() { m_window.clear(sf::Color::Black); }
void SimpleWindow::endDraw() { m_window.display(); }

void SimpleWindow::draw(const sf::Drawable& l_drawable) { m_window.draw(l_drawable); }

bool SimpleWindow::isOpen() const { return m_isOpen; }
bool SimpleWindow::isFullscreen() const { return m_isFullscreen; }
const sf::Vector2u& SimpleWindow::getWindowSizeU() const { return m_uWindowSize; }
const sf::Vector2f& SimpleWindow::getWindowSizeF() const { return m_fWindowSize; }
unsigned int SimpleWindow::getFramerate() const { return m_framerate; }
sf::RenderTarget& SimpleWindow::getRenderTarget() { return m_window; }
sf::RenderWindow& SimpleWindow::get() { return m_window; }

