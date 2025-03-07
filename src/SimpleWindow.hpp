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

	void ProcessEvents(sf::View& mainView);

	void ToggleFullscreen();

	void BeginDraw();
	void EndDraw();
	void Draw(const sf::Drawable& l_drawable);

	void setFramerate(unsigned int l_limit);
	void setView(sf::View& v);

	bool isOpen() const;
	bool isFullscreen() const;
	const sf::Vector2u& getWindowSizeU() const;
	const sf::Vector2f& getWindowSizeF() const;
	unsigned int getFramerate() const;
	sf::RenderTarget& getRenderTarget();
	sf::RenderWindow& get();
	sf::View& getView();

private:
	void Setup(const std::string& l_title, const sf::Vector2u& l_size);
	void Destroy();
	void Create();

	sf::RenderWindow m_window;
	sf::View m_view;
	sf::Vector2u m_uWindowSize;
	sf::Vector2f m_fWindowSize;
	std::string m_windowTitle;
	bool m_isOpen;
	bool m_isFullscreen;
	unsigned int m_framerate;
};