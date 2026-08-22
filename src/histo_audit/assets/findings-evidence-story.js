(() => {
  const story = document.getElementById('learned-story');
  if (!story) return;

  const rows = [...story.querySelectorAll('[data-evidence-step]')];
  const reducedQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
  const desktopQuery = window.matchMedia('(min-width: 901px)');

  if (reducedQuery.matches || !('IntersectionObserver' in window)) {
    rows.forEach(row => row.classList.add('is-mobile-visible'));
  } else {
    story.classList.add('findings-story--mobile-reveal');
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) entry.target.classList.add('is-mobile-visible');
      });
    }, {rootMargin: '0px 0px -12% 0px', threshold: .14});
    rows.forEach(row => observer.observe(row));
  }

  const gsapEngine = window.gsap;
  const scrollEngine = window.ScrollTrigger;
  if (!gsapEngine || !scrollEngine) return;
  gsapEngine.registerPlugin(scrollEngine);

  const media = gsapEngine.matchMedia();
  media.add('(min-width: 901px) and (prefers-reduced-motion: no-preference)', () => {
    const nodes = [...story.querySelectorAll('[data-evidence-node]')];
    const progressLine = story.querySelector('.evidence-spine-progress');
    if (rows.length !== 7 || nodes.length !== 7 || !progressLine) return;

    story.classList.add('findings-story--enhanced');
    const rowOffset = 320;
    const opacityFor = (index, active) => {
      const distance = index - active;
      if (distance === 0) return 1;
      if (distance === -1) return .28;
      if (distance === 1) return .14;
      return 0;
    };

    gsapEngine.set(rows, {
      yPercent: -50,
      y: index => index * rowOffset,
      autoAlpha: index => opacityFor(index, 0),
    });
    gsapEngine.set(nodes, {
      fill: index => index === 0 ? '#fff' : '#010102',
      stroke: '#fff',
      opacity: index => index === 0 ? 1 : .28,
    });
    gsapEngine.set(progressLine, {scaleY: 1 / 7, transformOrigin: '50% 0%'});

    const timeline = gsapEngine.timeline({paused: true, defaults: {ease: 'sine.inOut'}});
    for (let active = 1; active < 7; active += 1) {
      timeline.to(rows, {
        y: index => (index - active) * rowOffset,
        autoAlpha: index => opacityFor(index, active),
        duration: .88,
      }, active);
      timeline.to(nodes, {
        fill: index => index === active ? '#fff' : '#010102',
        stroke: '#fff',
        opacity: index => index === active ? 1 : (index < active ? .46 : .24),
        duration: .76,
      }, active);
      timeline.to(progressLine, {
        scaleY: (active + 1) / 7,
        duration: .82,
      }, active);
    }

    const trigger = scrollEngine.create({
      trigger: story,
      start: 'top top',
      end: 'bottom bottom',
      animation: timeline,
      scrub: .65,
      invalidateOnRefresh: true,
      anticipatePin: 0,
    });

    const refresh = () => scrollEngine.refresh();
    document.fonts?.ready.then(refresh);
    window.addEventListener('load', refresh, {once: true});
    window.addEventListener('resize', refresh, {passive: true});

    return () => {
      window.removeEventListener('resize', refresh);
      trigger.kill();
      timeline.kill();
      story.classList.remove('findings-story--enhanced');
      gsapEngine.set([...rows, ...nodes, progressLine], {
        clearProps: 'all',
      });
    };
  });

  const refreshForMediaChange = () => {
    if (desktopQuery.matches && !reducedQuery.matches) scrollEngine.refresh();
  };
  desktopQuery.addEventListener('change', refreshForMediaChange);
  reducedQuery.addEventListener('change', refreshForMediaChange);
})();
