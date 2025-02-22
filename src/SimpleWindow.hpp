#pragma once
#include <string>
#include <SFML/Graphics/Drawable.hpp>
#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/View.hpp>
#include <SFML/System/Vector2.hpp>

class SimpleWindow
{
public:
	SimpleWindow();
	SimpleWindow(const std::string& l_title, const sf::Vector2u& l_size);
	~SimpleWindow();

	void processEvents();

	void toggleFullscreen();
	void setFramerate(unsigned int l_limit);

	void beginDraw();
	void endDraw();

	void draw(const sf::Drawable& l_drawable);

	bool isOpen() const;
	bool isFullscreen() const;
	sf::Vector2u getWindowSize() const;
	unsigned int getFramerate() const;
	sf::RenderTarget& getRenderTarget();
	sf::RenderWindow& get();

private:
	void setup(const std::string& l_title, const sf::Vector2u& l_size);
	void destroy();
	void create();

	sf::RenderWindow m_window;
	sf::Vector2u m_windowSize;
	std::string m_windowTitle;
	bool m_isOpen;
	bool m_isFullscreen;
	unsigned int m_framerate;
};