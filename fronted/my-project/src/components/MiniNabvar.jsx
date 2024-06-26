import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { links } from "../data";

function MiniNavbar() {
    const filteredLinks = links.filter((link) => link.text !== 'Log in');

    return (
        <nav className="bg-emerald-100">
            <div className="align-element py-4 flex flex-col sm:flex-row sm:gap-x-16 sm:items-center sm:py-8">
                <h2 className="text-3xl font-bold">
                    Order<span className="text-emerald-600">Boost</span>
                </h2>
                <div className="flex gap-x-3">
                    {filteredLinks.map((link) => {
                        const { id, href, text } = link;
                        return (
                            <a key={id} href={href} className="capitalized text-lg tracking-wide hover:text-emerald-600 duration-300">
                                {text}
                            </a>
                        );
                    })}
                </div>
            </div>
        </nav>
    );
}

export default MiniNavbar;
