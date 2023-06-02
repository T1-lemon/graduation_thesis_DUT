import {
  Avatar,
  Box,
  ClickAwayListener,
  Grow,
  Paper,
  Popper,
  Typography
} from '@mui/material'
import React, { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { MenuSettingsInteface } from '../../interface/HeaderInterface'
import { useAppDispatch, useAppSelector } from '../../redux/hooks'
import { logoutUser } from '../../redux/thunk/authThunk'
import { MenuItemOverride, MenuListOverrride } from '../../themes/headerTheme'
import { settings } from '../../utils/HeaderSetting'
import ButtonCustomize from '../ButtonCustomize'

type Props = {
  settings: MenuSettingsInteface[]
}

const MenuHeader = (props: Props) => {
  const { settings } = props
  const navigate = useNavigate()

  const { isAuth, currentUser } = useAppSelector((state) => state.auths)
  const dispatch = useAppDispatch()

  const [open, setOpen] = useState(false)
  const buttonRef = useRef<HTMLButtonElement>(null)

  const handleToggle = () => {
    setOpen((prevOpen) => !prevOpen)
  }

  const handleLogoutUser = async () => {
    await dispatch(logoutUser())
    navigate('/home')
  }

  const handleClose = (
    event: Event | React.SyntheticEvent,
    link?: string,
    action?: boolean
  ) => {
    if (link) {
      navigate(`${link}`)
    }

    if (action) {
      handleLogoutUser()
    }

    if (
      buttonRef.current &&
      buttonRef.current.contains(event.target as HTMLElement)
    ) {
      return
    }

    setOpen(false)
  }

  function handleListKeyDown(event: React.KeyboardEvent) {
    if (event.key === 'Tab') {
      event.preventDefault()
      setOpen(false)
    } else if (event.key === 'Escape') {
      setOpen(false)
    }
  }

  const prevOpen = useRef(open)
  useEffect(() => {
    if (prevOpen.current === true && open === false) {
      buttonRef.current!.focus()
    }

    prevOpen.current = open
  }, [open])
  return (
    <Box>
      {isAuth ? (
        <div>
          <button
            ref={buttonRef}
            onClick={handleToggle}
            className='header__button--avatar'
          >
            <Avatar
              alt={currentUser.lastName}
              src={currentUser.profilePicture}
              className={open ? 'header__avatar--border' : 'header__avatar'}
            />
            <Typography className='header__avatar--username'>
              {currentUser.lastName}
            </Typography>
          </button>
          <Popper
            open={open}
            anchorEl={buttonRef.current}
            role={undefined}
            placement='bottom-start'
            transition
            disablePortal
            sx={{ zIndex: '2' }}
          >
            {({ TransitionProps, placement }) => (
              <Grow
                {...TransitionProps}
                style={{
                  transformOrigin:
                    placement === 'bottom-start' ? 'left top' : 'left bottom'
                }}
              >
                <Paper>
                  <ClickAwayListener onClickAway={handleClose}>
                    <MenuListOverrride
                      autoFocusItem={open}
                      id='composition-menu'
                      aria-labelledby='composition-button'
                      onKeyDown={handleListKeyDown}
                    >
                      {settings.map((setting) => (
                        <MenuItemOverride
                          key={setting.menuItem}
                          onClick={(event) =>
                            handleClose(event, setting.link, setting.action)
                          }
                        >
                          {setting.link ? (
                            <Typography textAlign='center'>
                              {setting.menuItem}
                            </Typography>
                          ) : (
                            <Typography textAlign='center'>
                              {setting.menuItem}
                            </Typography>
                          )}
                        </MenuItemOverride>
                      ))}
                    </MenuListOverrride>
                  </ClickAwayListener>
                </Paper>
              </Grow>
            )}
          </Popper>
        </div>
      ) : (
        <ButtonCustomize
          text='Sign in'
          className='btn__radius'
          onClickBtn={() => navigate('/login')}
        />
      )}
    </Box>
  )
}

export default MenuHeader
